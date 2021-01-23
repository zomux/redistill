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
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.progressive:
            random_refine_step = sample["random_refine_step"]
            model.decoder.select_decoder(random_refine_step)
        net_output = model(**sample['net_input'])
        loss, reward, sample_reward, avgkl = self.compute_loss(model, net_output, sample, reduce=reduce)
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

    def compute_loss(self, model, net_output, sample, reduce=True):
        y_input = sample["net_input"]["prev_output_tokens"]
        y_mask = (y_input != 1)
        cmlm_mask = (y_input == 4)
        true_target = sample["true_target"]
        logits = net_output[0]
        # >>>>>> Compute policy >>>>>>>>
        probs = torch.softmax(logits / self.args.sampling_temp, 2).clamp(min=1e-8, max=1.)
        log_probs = torch.log(probs)
        y_pred = logits.argmax(2)
        y_pred = y_pred * cmlm_mask + y_input * cmlm_mask.logical_not()
        # >>>>>> Compute baseline >>>>>>>>
        with torch.no_grad():
            baseline_logits = self.task.baseline_model()(**sample['net_input'])[0]
            # baseline_log_probs = torch.log_softmax(baseline_logits / self.args.sampling_temp, 2)
            baseline_ypred = baseline_logits.argmax(2)
            baseline_ypred = baseline_ypred * cmlm_mask + y_input * cmlm_mask.logical_not()
        # >>>>>> Obtain sample >>>>>>>>
        B, T, _ = probs.shape
        y_sample = torch.multinomial(probs.view(B*T, -1), 1).view(B, T)
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
        kldiv = cmlm_mask.sum(1).float() * 0.
        if self.args.tokenwise:
            reward = reward / cmlm_mask.sum(1)
            sample_reward = sample_reward / cmlm_mask.sum(1)
            loss = (- sample_reward[:, None] * logp_mat * cmlm_mask).sum(1) / cmlm_mask.sum(1) # + self.args.klreg * kldiv / cmlm_mask.sum(1)
        else:
            logp = (logp_mat * cmlm_mask).sum(1)
            loss = - sample_reward * logp # + self.args.klreg * kldiv
        avgkl = kldiv / cmlm_mask.sum(1)
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