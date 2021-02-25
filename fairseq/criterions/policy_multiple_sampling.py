import math
import numpy as np
from fairseq import utils

import math
import torch
from . import FairseqCriterion, register_criterion
from .lib_sbleu import smoothed_bleu

POOL_SIZE = 20000
SHARD_SIZE = 32

@register_criterion('policy_multiple_sampling')
class PolicyMultipleSampling(FairseqCriterion):

    def __init__(self, args, task):
        self.pool = []
        self.progressive = hasattr(args, "progressive") and args.progressive
        self.masker = hasattr(args, "masker") and args.masker
        self.exposure1 = args.exposure1
        self.tgt_dict = task.tgt_dict
        self.args = args
        self.training_cnt = 0
        if args.exposure1 or True:
            from fairseq.strategies.mask_predict import MaskPredict
            args.end_iteration = -1
            args.decoding_iterations = args.refinetot
            self.strategy = MaskPredict(args, exit_after_mask=False if self.masker or getattr(args, "pnet2", False) else True)
        else:
            self.strategy = None
        assert not getattr(self.args, "progressive", False)
        super().__init__(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--klreg', default=0, type=float)
        parser.add_argument("--sampling-temp", default=0.5, type=float)
        parser.add_argument("--tokenwise", action="store_true")
        parser.add_argument("--exposure1", action="store_true", help="refine during training")
        # fmt: on

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
        y_input = net_input["prev_output_tokens"]
        net_output = model.decoder(y_input, encoder_out=encoder_out)
        loss, reward, relative_reward, absreward = self.compute_loss(model, encoder_out, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'reward': utils.item(reward.data) if reduce else reward.data,
            'relative_reward': utils.item(relative_reward.data) if reduce else relative_reward.data,
            'absreward': utils.item(absreward.data) if reduce else absreward.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if model.training:
            self.training_cnt += 1
        return loss, sample_size, logging_output

    def compute_reward(self, y_sample, true_target):
        return self._sbleu(y_sample, true_target)

    def compute_loss(self, model, encoder_out, net_output, sample, reduce=True):
        y_input = sample["net_input"]["prev_output_tokens"]
        y_mask = (y_input != 1)
        cmlm_mask = (y_input == 4)
        true_target = sample["true_target"]
        logits = net_output[0]

        # >>>>>> Compute policy probs >>>>>>>>
        probs = torch.softmax(logits, 2).clamp(min=1e-8, max=1.)
        probs = probs.clamp(min=1e-8, max=1.)
        B, T, _ = probs.shape
        sample_sz = 10
        samples = torch.multinomial(probs.view(B*T, -1), sample_sz, replacement=True).view(B, T, sample_sz).transpose(1, 2)
        samples = samples * y_mask[:, None, :] + samples.clone().fill_(1) * y_mask.logical_not()[:, None, :]
        samples = samples * cmlm_mask[:, None, :] + (y_input * cmlm_mask.logical_not())[:, None, :]

        # >>>>> Compute reward >>>>>>>>
        flat_samples = (samples * y_mask[:, None, :]).view(B * sample_sz, -1)
        flat_targets = (true_target * y_mask)[:, None, :].repeat(1, sample_sz, 1).view(B * sample_sz, -1)
        rewards = self.compute_reward(flat_samples, flat_targets).view(B, sample_sz)
        best_idx = rewards.argmax(1)
        best_samples = samples[torch.arange(B), best_idx]
        pred = logits.argmax(2)
        pred_rewards = self.compute_reward(pred * y_mask, true_target * y_mask)

        # >>>> Compute loss >>>>>>>>>>
        log_probs = torch.log(probs)
        logp_mat = log_probs.view(B*T, -1)[torch.arange(B*T), best_samples.flatten()].view(B, T)
        loss = - (logp_mat * cmlm_mask).sum(1) / (cmlm_mask.sum(1) + 1e-8)
        if self.args.klreg > 0:
            with torch.no_grad():
                baseline_logits, _ = self.task.baseline_model().decoder(y_input,encoder_out=encoder_out)
                baseline_log_probs = torch.log_softmax(baseline_logits, 2)
            kldiv_mat = (probs * log_probs / baseline_log_probs).sum(2) * cmlm_mask
            kldiv = kldiv_mat.sum(1) / (cmlm_mask.sum(1) + 1e-8)
            loss += - self.args.klreg * kldiv

        absreward = pred_rewards
        reward = rewards.max(1).values
        relative_reward = reward - absreward

        if reduce:
            loss = loss.sum()
            reward = reward.sum()
            relative_reward = relative_reward.sum()
            absreward = absreward.sum()
        return loss, reward, relative_reward, absreward

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'reward': sum(log.get('reward', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'relative_reward': sum(log.get('relative_reward', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'absreward': sum(log.get('absreward', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
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