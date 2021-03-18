# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import numpy as np
import math
import sys

from fairseq import utils

from . import FairseqCriterion, register_criterion
from .lib_sbleu import smoothed_bleu

SHARD_SIZE = 20


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('reward_cross_entropy')
class RewardCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        from fairseq.sequence_generator import SequenceGenerator
        self.gen = SequenceGenerator(task.target_dictionary, beam_size=args.beam_size)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--proxyloss2', action="store_true")
        parser.add_argument('--beam-size', type=int, default=4)
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # >>>> Sample for reward >>>
        is_training = model.training
        model.eval()
        B = sample["target"].shape[0]
        gen_results = []
        for shard_i in range(math.ceil(float(B) / SHARD_SIZE)):
            start = shard_i * SHARD_SIZE
            end = (shard_i + 1) * SHARD_SIZE
            sub_sample = {
                "net_input": {
                    "src_tokens": sample["net_input"]["src_tokens"][start:end],
                    "src_lengths": sample["net_input"]["src_lengths"][start:end],
                    "prev_output_tokens": sample["net_input"]["prev_output_tokens"][start:end]
                }
            }
            sub_results = [[p["tokens"][:60] for p in results] for results in self.gen.generate([model], sub_sample)]
            gen_results.extend(sub_results)
        targets = sample["target"] * torch.gt(sample["target"], 1)
        rewards = []
        for batch_i in range(len(gen_results)):
            batch_rewards = []
            for seq in gen_results[batch_i]:
                batch_rewards.append(self.compute_reward(seq, targets[batch_i]))
            rewards.append(batch_rewards)
        rewards = torch.tensor(rewards)
        best_idx = rewards.argmax(1)
        best_results = [res[idx] for res, idx in zip(gen_results, best_idx)]
        if not self.args.proxyloss2:
            maxlen = max([len(r) for r in best_results])
            new_target = targets.new_ones(targets.shape[0], maxlen)
            for i, seq in enumerate(best_results):
                new_target[i, :seq.shape[0]] = seq
            first_col = new_target.new_ones(new_target.shape[0]) * 2
            new_decoder_input = torch.cat([ first_col[:, None], new_target[:, :-1] ], 1)
        else:
            # argmax_results = [res[0] for res in gen_results]
            worst_results = [res[idx] for res, idx in zip(gen_results, rewards.argmin(1))]
            merged_results = best_results + worst_results
            maxlen = max([len(r) for r in merged_results])
            new_target = targets.new_ones(len(merged_results), maxlen)
            for i, seq in enumerate(merged_results):
                new_target[i, :seq.shape[0]] = seq
            first_col = new_target.new_ones(new_target.shape[0]) * 2
            new_decoder_input = torch.cat([ first_col[:, None], new_target[:, :-1] ], 1)
        sample["net_input"]["prev_output_tokens"] = new_decoder_input
        sample["target"] = new_target
        best_reward = rewards[torch.arange(rewards.shape[0]), best_idx].cuda()
        argmax_reward = rewards[:, 0].cuda()
        if is_training:
            model.train()
        # >>>>
        if not self.args.proxyloss2:
            decoder_out = model.forward(sample['net_input']["src_tokens"], sample['net_input']["src_lengths"], new_decoder_input)
            loss, nll_loss = self.compute_loss(model, decoder_out, sample, reduce=reduce)
        else:
            # repeated_encoder_out = {"encoder_out": torch.cat([encoder_out["encoder_out"], encoder_out["encoder_out"]]),
            #                         "encoder_padding_mask":
            #                             torch.cat([encoder_out["encoder_padding_mask"], encoder_out["encoder_padding_mask"]])
            #                             if encoder_out["encoder_padding_mask"] is not None else None
            #                         }
            repeated_src_tokens = torch.cat([sample['net_input']["src_tokens"], sample['net_input']["src_tokens"]])
            repeated_src_lengths = torch.cat([sample['net_input']["src_lengths"], sample['net_input']["src_lengths"]])
            decoder_out = model.forward(repeated_src_tokens, repeated_src_lengths, new_decoder_input)
            loss, nll = self.compute_loss(model, decoder_out, {"target": new_target}, reduce=False, return_full_mat=True)
            token_mask = torch.ne(new_target, self.padding_idx)
            loss = (loss.view(new_target.shape) * token_mask).sum(1) / token_mask.sum(1)
            nll = (nll.view(new_target.shape) * token_mask).sum(1) / token_mask.sum(1)
            loss = (loss[:B] - loss[-B:]) * 10. + 0.1
            nll_loss = (nll[:B] - nll[-B:]) * 10.
            loss = torch.gt(loss, 0) * loss
            if reduce:
                loss = loss.sum()
                nll_loss = nll_loss.sum()
        sample_size = B if self.args.sentence_avg or self.args.proxyloss2 else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'best_r': utils.item(best_reward.sum().data) if reduce else best_reward.data,
            'argmax_r': utils.item(argmax_reward.sum().data) if reduce else argmax_reward.data,
            'ntokens': sample['ntokens'],
            'nsentences': B,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, return_full_mat=False):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        if return_full_mat:
            ignore_index = None
        else:
            ignore_index = self.padding_idx
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=ignore_index, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'best_r': sum(log.get('best_r', 0) for log in logging_outputs) / nsentences / math.log(2) if nsentences > 0 else 0.,
            'argmax_r': sum(log.get('argmax_r', 0) for log in logging_outputs) / nsentences / math.log(2) if nsentences > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

    def compute_reward(self, yhat, target):
        return self._sbleu(yhat, target)

    def _sbleu(self, yhat, target):
        tgt_seq = target.int().cpu().numpy()
        sampled_tokens = yhat.int().cpu().numpy()
        tgt_mask = np.greater(tgt_seq, 0)
        yhat_mask = np.greater(sampled_tokens, 0)
        target_len = int(tgt_mask.sum())
        yhat_len = int(yhat_mask.sum())
        ref_tokens = tgt_seq[:target_len]
        out_tokens = list(sampled_tokens[:yhat_len])
        ref_tokens = self.task.tgt_dict.string(ref_tokens).replace("@@ ", "").split()
        out_tokens = self.task.tgt_dict.string(out_tokens).replace("@@ ", "").split()
        return smoothed_bleu(out_tokens, ref_tokens)
