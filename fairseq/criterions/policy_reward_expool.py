import sys
import math
import numpy as np
from fairseq import utils

import torch
from . import FairseqCriterion, register_criterion
from .lib_sbleu import smoothed_bleu
from fairseq import distributed_utils

POOL_SIZE = 20000
REWARD_HIST_SIZE = 50

@register_criterion('policy_reward')
class PolicyRewardExPoolCriterion(FairseqCriterion):

    def __init__(self, args, task):
        global POOL_SIZE
        self.pool = []
        self.stable_reward = args.stablereward
        self.reward_history = []
        self.progressive = hasattr(args, "progressive") and args.progressive
        if self.progressive:
            POOL_SIZE = 80000
        super().__init__(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--klreg', default=0, type=float)
        parser.add_argument("--sampling-temp", default=0.5, type=float)
        parser.add_argument("--offpolicy", action="store_true")
        parser.add_argument("--tokenwise", action="store_true")
        parser.add_argument("--stablereward", action="store_true")
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert self.args.offpolicy
        # sample_size = sample["net_input"]["src_tokens"].size(0)
        if not model.training:
            pool = []
            sample_size = sample["net_input"]["src_tokens"].size(0)
        else:
            pool = self.pool
            sample_size = 200

        self.accumulate_expool(sample, pool)
        if self.progressive:
            random_refine_step = sample["random_refine_step"]
        else:
            random_refine_step = 0
        sample = self.sample_expool(sample_size, pool, random_reifne_step=random_refine_step)
        if self.progressive:
            model.decoder.select_decoder(random_refine_step)
        net_output = model(**sample["net_input"])
        loss, reward, sample_reward = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample["true_targets"].size(0)
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'reward': utils.item(reward.data) if reduce else reward.data,
            'sample_reward': utils.item(sample_reward.data) if reduce else sample_reward.data,
            'ntokens': int((sample["true_targets"] != 1).long().sum()),
            'nsentences': sample_size,
            'sample_size': sample_size,
            'pool_size': len(self.pool),
        }
        return loss, sample_size, logging_output

    def compute_reward(self, y_sample, true_target):
        return self._sbleu(y_sample, true_target)

    def sample_expool(self, size, pool, random_reifne_step=0):
        size = min(size, len(pool))
        if not self.progressive:
            id_pool = np.arange(len(pool))
        else:
            id_pool = np.array([i for i in range(len(pool)) if pool[i][-1] == random_reifne_step])
            size = min(size, len(id_pool))
        ids = self.task.random.choice(id_pool, size, replace=False)
        if len(ids) == 0:
            from lib_nsmldebug import set_trace
            set_trace()
        max_xlen = max(pool[i][0].shape[0] for i in ids)
        max_ylen = max(pool[i][1].shape[0] for i in ids)
        if max_ylen * len(ids) > self.args.max_tokens:
            ids = ids[:int(self.args.max_tokens / max_ylen)]
        B = len(ids)
        src_tokens = torch.cuda.LongTensor(B, max_xlen).fill_(1)
        src_lengths = torch.cuda.LongTensor(B)
        tgt_inputs = torch.cuda.LongTensor(B, max_ylen).fill_(1)
        tgt_samples = torch.cuda.LongTensor(B, max_ylen).fill_(1)
        baseline_logp = torch.cuda.FloatTensor(B, max_ylen).fill_(0.)
        true_targets = torch.cuda.LongTensor(B, max_ylen).fill_(1)
        sample_rewards = torch.cuda.FloatTensor(B)
        baseline_rewards = torch.cuda.FloatTensor(B)
        for i in range(B):
            id = ids[i]
            src_token, tgt_input, tgt_sample, true_target, logp, sample_reward, baseline_reward, _ = pool[id]
            src_tokens[i, :len(src_token)] = src_token
            src_lengths[i] = len(src_token)
            tgt_inputs[i, :len(tgt_input)] = tgt_input
            tgt_samples[i, :len(tgt_input)] = tgt_sample
            baseline_logp[i, :len(tgt_input)] = logp
            true_targets[i, :len(true_target)] = true_target
            sample_rewards[i] = sample_reward
            baseline_rewards[i] = baseline_reward
        sample = {
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "prev_output_tokens": tgt_inputs
            },
            "true_targets": true_targets,
            "target_samples": tgt_samples,
            "sample_rewards": sample_rewards,
            "baseline_rewards": baseline_rewards,
            "baseline_logp": baseline_logp
        }
        return sample

    def accumulate_expool(self, sample, pool):
        y_input = sample["net_input"]["prev_output_tokens"]
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        y_mask = (y_input != 1)
        cmlm_mask = (y_input == 4)
        true_target = sample["true_target"]
        tgt_lengths = (true_target == 2).nonzero()[:, 1] + 1
        if "random_refine_step" in sample:
            random_refine_step = sample["random_refine_step"]
        else:
            random_refine_step = 0
        with torch.no_grad():
            baseline_logits = self.task.baseline_model()(**sample['net_input'])[0]
            baseline_log_probs = torch.log_softmax(baseline_logits / self.args.sampling_temp, 2)
            baseline_ypred = baseline_logits.argmax(2)
            baseline_ypred = baseline_ypred * cmlm_mask + y_input * cmlm_mask.logical_not()
            baseline_probs = torch.exp(baseline_log_probs)
            B, T, _ = baseline_probs.shape
            y_sample = torch.multinomial(baseline_probs.view(B * T, -1), 1).view(B, T)
            y_sample = y_sample * cmlm_mask + y_input * cmlm_mask.logical_not()
            logp_mat = baseline_log_probs.view(B*T, -1)[torch.arange(B*T), y_sample.flatten()].view(B, T)
            sample_reward = self.compute_reward(y_sample * y_mask, true_target * y_mask).cpu()
            baseline_reward = self.compute_reward(baseline_ypred * y_mask, true_target * y_mask).cpu()
            relative_reward = sample_reward - baseline_reward
        for i in range(y_input.shape[0]):
            if relative_reward[i] != 0.:
                src_token = src_tokens[i, :src_lengths[i]].cpu()
                tgt_input = y_input[i, :tgt_lengths[i]].cpu()
                tgt_sample = y_sample[i, :tgt_lengths[i]].cpu()
                target = true_target[i, :tgt_lengths[i]].cpu()
                logp = logp_mat[i, :tgt_lengths[i]].cpu()
                pool.append((src_token, tgt_input, tgt_sample, target, logp, sample_reward[i], baseline_reward[i], random_refine_step))
        while len(pool) > POOL_SIZE:
            pool.pop(0)

    def compute_loss(self, model, net_output, sample, reduce=True):
        y_input = sample["net_input"]["prev_output_tokens"]
        y_mask = (y_input != 1)
        cmlm_mask = (y_input == 4)
        true_target = sample["true_targets"]
        logits = net_output[0]
        # >>>>>> Compute policy >>>>>>>>
        probs = torch.softmax(logits / self.args.sampling_temp, 2).clamp(min=1e-8, max=1.0)
        log_probs = torch.log(probs)
        y_pred = logits.argmax(2)
        y_pred = y_pred * cmlm_mask + y_input * cmlm_mask.logical_not()
        # >>>>>> Compute baseline >>>>>>>>
        sample_rewards = sample["sample_rewards"]
        baseline_rewards = sample["baseline_rewards"]
        # >>>>>> Obtain sample >>>>>>>>
        y_sample = sample["target_samples"]
        B, T, _ = log_probs.shape
        logp_mat = log_probs.view(B*T, -1)[torch.arange(B*T), y_sample.flatten()].view(B, T)
        # >>>>>> Compute reward >>>>>>>>
        reward = self.compute_reward(y_pred * y_mask, true_target * y_mask) - baseline_rewards
        sample_reward = sample_rewards - baseline_rewards
        baseline_logp_mat = sample["baseline_logp"]
        # >>>>>> reject outliers >>>>>>
        if self.stable_reward and model.training:
            reward_mean = float(sample_reward.mean())
            self.reward_history.append(reward_mean)
            if len(self.reward_history) > REWARD_HIST_SIZE:
                self.reward_history.pop(0)
            if len(self.reward_history) == REWARD_HIST_SIZE:
                hist_mean = np.mean(self.reward_history)
                hist_std = np.std(self.reward_history)
                if reward_mean < hist_mean - 2 * hist_std or reward_mean > hist_mean + 2 * hist_std:
                    sample_reward.fill_(0.)
        # >>>>>> Compute loss signal >>>>>>>>
        if self.args.tokenwise:
            with torch.no_grad():
                importance_weight = torch.exp(logp_mat - baseline_logp_mat)
            logp_mat = importance_weight * logp_mat
            reward = reward / cmlm_mask.sum(1)
            sample_reward = sample_reward / cmlm_mask.sum(1)
            loss = (- sample_reward[:, None] * logp_mat * cmlm_mask).sum(1) / cmlm_mask.sum(1)
        else:
            logp = (logp_mat * cmlm_mask).sum(1)
            with torch.no_grad():
                importance_weight = torch.exp(((logp_mat - baseline_logp_mat) * cmlm_mask).sum(1))
            logp = importance_weight * logp
            loss = - sample_reward * logp
        if reduce:
            loss = loss.sum()
            reward = reward.sum()
            sample_reward = sample_reward.sum()
        return loss, reward, sample_reward

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
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'pool_size': sum(log.get('pool_size', 0) for log in logging_outputs) / len(logging_outputs),
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