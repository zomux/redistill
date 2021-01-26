import math
import numpy as np
from fairseq import utils

import torch
from . import FairseqCriterion, register_criterion
from .lib_sbleu import smoothed_bleu

POOL_SIZE = 20000

@register_criterion('policy_reward_onpolicy')
class PolicyRewardCriterion(FairseqCriterion):

    def __init__(self, args, task):
        self.pool = []
        self.progressive = hasattr(args, "progressive") and args.progressive
        self.masker = hasattr(args, "masker") and args.masker
        self.exposure1 = args.exposure1
        self.tgt_dict = task.tgt_dict
        if args.exposure1:
            from fairseq.strategies.mask_predict import MaskPredict
            args.end_iteration = -1
            args.decoding_iterations = args.refinetot
            self.strategy = MaskPredict(args, exit_after_mask=False if self.masker else True)
        else:
            self.strategy = None
        assert not args.offpolicy
        super().__init__(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--klreg', default=0, type=float)
        parser.add_argument("--sampling-temp", default=0.5, type=float)
        parser.add_argument("--offpolicy", action="store_true")
        parser.add_argument("--tokenwise", action="store_true")
        parser.add_argument("--exposure1", action="store_true", help="refine during training")
        # fmt: on

    def perform_refinement(self, model, sample, encoder_out):
        assert "random_refine_step" in sample
        random_refine_step = sample["random_refine_step"]
        was_training = model.training
        model.eval()
        y_input = sample["net_input"]["prev_output_tokens"]
        y_mask = (y_input != 1)
        y_input = (y_mask * 0 + 4) * y_mask + y_input * y_mask.logical_not()
        if random_refine_step == 0:
            return y_input
        with torch.no_grad():
            self.strategy.end_iteration = random_refine_step - 1
            out_tokens, _ = self.strategy.generate(model, encoder_out, y_input, self.tgt_dict)
        if was_training:
            model.train()
        return out_tokens

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_input = sample["net_input"]
        with torch.no_grad():
            encoder_out = model.encoder(net_input["src_tokens"], src_lengths=net_input["src_lengths"])
        if self.exposure1:
            y_input = self.perform_refinement(model, sample, encoder_out)
            sample["net_input"]["prev_output_tokens"] = y_input
        else:
            y_input = net_input["prev_output_tokens"]
        if self.progressive:
            random_refine_step = sample["random_refine_step"]
            model.decoder.select_decoder(random_refine_step)
        net_output = model.decoder(y_input, encoder_out=encoder_out)
        loss, reward, sample_reward, avgkl = self.compute_loss(model, encoder_out, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'reward': utils.item(reward.data) if reduce else reward.data,
            'sample_reward': utils.item(sample_reward.data) if reduce else sample_reward.data,
            'avgkl': utils.item(avgkl.data) if reduce else avgkl.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_reward(self, y_sample, true_target):
        return self._sbleu(y_sample, true_target)

    def compute_loss(self, model, encoder_out, net_output, sample, reduce=True):
        y_input = sample["net_input"]["prev_output_tokens"]
        y_mask = (y_input != 1)
        cmlm_mask = (y_input == 4)
        true_target = sample["true_target"]
        logits = net_output[0]
        # >>>>>> Compute policy >>>>>>>>
        probs = torch.softmax(logits / self.args.sampling_temp, 2).clamp(min=1e-8, max=1.)
        if self.masker and model.decoder.selected_decoder > 0:
            input_onehot = torch.nn.functional.one_hot(y_input, logits.shape[2])
            masking_prob = net_output[1]["masking_prob"]
            probs = masking_prob[:, :, None] * input_onehot + (1 - masking_prob[:, :, None]) * probs
        log_probs = torch.log(probs)
        y_pred = logits.argmax(2)
        if self.masker:
            y_pred = y_pred * y_mask
        else:
            y_pred = y_pred * cmlm_mask + y_input * cmlm_mask.logical_not()
        # >>>>>> Compute baseline >>>>>>>>
        if self.masker and model.decoder.selected_decoder > 0:
            baseline_ypred = y_input.detach()
        else:
            with torch.no_grad():
                baseline_logits = self.task.baseline_model().decoder(y_input, encoder_out=encoder_out)[0]
                # baseline_log_probs = torch.log_softmax(baseline_logits / self.args.sampling_temp, 2)
                baseline_ypred = baseline_logits.argmax(2)
                baseline_ypred = baseline_ypred * cmlm_mask + y_input * cmlm_mask.logical_not()

        # >>>>>> Obtain sample >>>>>>>>
        B, T, _ = probs.shape
        y_sample = torch.multinomial(probs.view(B*T, -1), 1).view(B, T)
        if not self.masker:
            y_sample = y_sample * cmlm_mask + y_input * cmlm_mask.logical_not()
        logp_mat = log_probs.view(B*T, -1)[torch.arange(B*T), y_sample.flatten()].view(B, T)
        # >>>>>> Compute reward >>>>>>>>
        sample_reward = self.compute_reward(y_sample * y_mask, true_target * y_mask)
        baseline_reward = self.compute_reward(baseline_ypred * y_mask, true_target * y_mask)
        reward = self.compute_reward(y_pred * y_mask, true_target * y_mask) - baseline_reward
        sample_reward = sample_reward - baseline_reward
        # >>>>>> Compute loss signal >>>>>>>>
        # >>>>>> Only for on-policy training >>>>>>>>
        # kldiv = ((probs * (log_probs - baseline_log_probs)).sum(2) * cmlm_mask).sum(1)
        kldiv = baseline_reward if self.masker else cmlm_mask.sum(1).float() * 0.
        if self.args.tokenwise:
            reward = reward / cmlm_mask.sum(1)
            sample_reward = sample_reward / cmlm_mask.sum(1)
            loss = (- sample_reward[:, None] * logp_mat * cmlm_mask).sum(1) / cmlm_mask.sum(1) # + self.args.klreg * kldiv / cmlm_mask.sum(1)
        else:
            if self.masker:
                logp = (logp_mat * y_mask).sum(1)
            else:
                logp = (logp_mat * cmlm_mask).sum(1)
            loss = - sample_reward * logp # + self.args.klreg * kldiv
        avgkl = baseline_reward if self.masker else kldiv / cmlm_mask.sum(1)
        if reduce:
            loss = loss.sum()
            reward = reward.sum()
            sample_reward = sample_reward.sum()
            avgkl = avgkl.sum()
        return loss, reward, sample_reward, avgkl

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'reward': sum(log.get('reward', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'sample_reward': sum(log.get('sample_reward', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'avgkl': sum(log.get('avgkl', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

    def _sbleu(self, yhat, target):
        bleus = []
        tgt_seq = target.int().cpu().numpy()
        sampled_tokens = yhat.int().cpu().numpy()
        tgt_mask = np.greater(tgt_seq, 0)
        yhat_mask = np.greater(sampled_tokens, 0)
        for i in range(tgt_seq.shape[0]):
            target_len = int(tgt_mask[i].sum())
            yhat_len = int(yhat_mask[i].sum())
            ref_tokens = tgt_seq[i, :target_len]
            out_tokens = list(sampled_tokens[i, :yhat_len])
            ref_tokens = self.task.tgt_dict.string(ref_tokens).replace("@@ ", "").split()
            out_tokens = self.task.tgt_dict.string(out_tokens).replace("@@ ", "").split()
            if not out_tokens:
                bleus.append(0.)
            else:
                bleus.append(smoothed_bleu(out_tokens, ref_tokens))
        return yhat.new_tensor(bleus, dtype=float)