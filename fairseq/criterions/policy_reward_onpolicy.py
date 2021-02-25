import math
import numpy as np
from fairseq import utils

import math
import torch
from . import FairseqCriterion, register_criterion
from .lib_sbleu import smoothed_bleu

POOL_SIZE = 20000
SHARD_SIZE = 32

@register_criterion('policy_reward_onpolicy')
class PolicyRewardCriterion(FairseqCriterion):

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
            if was_training:
                model.train()
            return y_input
        with torch.no_grad():
            if self.masker:
                out_tokens = y_input
                for i in range(random_refine_step):
                    input_tokens = out_tokens
                    model.decoder.select_decoder(i)
                    h, decoder_outs = model.decoder(input_tokens, encoder_out, compute_logits=False)
                    out_tokens_stack = []
                    for shard_i in range(math.ceil(float(h.shape[0]) / SHARD_SIZE)):
                        start = shard_i * SHARD_SIZE
                        end = (shard_i + 1) * SHARD_SIZE
                        shard_h = h[start:end]
                        logits = model.decoder.compute_logits(shard_h)
                        probs = torch.softmax(logits / self.args.sampling_temp, 2).clamp(min=1e-8, max=1.)
                        if i > 0:
                            masking_prob = decoder_outs["masking_prob"][start:end]
                            input_onehot = torch.nn.functional.one_hot(input_tokens[start:end], logits.shape[2])
                            probs = masking_prob[:, :, None] * probs + (1 - masking_prob[:, :, None]) * input_onehot
                            # probs = probs * (1 - masking_prob)[:, :, None]
                            # B, T, _ = probs.shape
                            # probs.view(B*T, _)[torch.arange(B * T), input_tokens[start:end].flatten()] += masking_prob.flatten()
                        out_tokens = probs.argmax(2)
                        out_tokens = out_tokens * y_mask[start:end] + input_tokens[start:end] * y_mask[start:end].logical_not()
                        out_tokens_stack.append(out_tokens)
                    out_tokens = torch.cat(out_tokens_stack, 0)
                    # if i == 1 and "5415,  460,  460, 4135, 4381,   18,  704, 1424" in str(input_tokens[-1]):
                    #     print("refp_in", input_tokens[-1])
                    #     # print("logits", logits[-1])
                    #     print("probs", probs[-1])
                    #     print("refp_out", out_tokens[-1])
            elif getattr(self.args, "pnet2", False):
                out_tokens = y_input
                for i in range(random_refine_step):
                    input_tokens = out_tokens
                    model.decoder.select_decoder(i)
                    h, decoder_padding_mask, decoder_out = model.decoder.compute_pnet_firsthalf(input_tokens, encoder_out)
                    pnet_out = decoder_out["pnet_out"]
                    mask = torch.sigmoid(pnet_out)
                    if i == 0:
                        mask = mask * 0.
                    last_h = model.decoder.compute_pnet_secondhalf(h, decoder_padding_mask, mask, encoder_out, compute_logits=False)
                    out_tokens_stack = []
                    for shard_i in range(math.ceil(float(h.shape[0]) / SHARD_SIZE)):
                        start = shard_i * SHARD_SIZE
                        end = (shard_i + 1) * SHARD_SIZE
                        shard_h = last_h[start:end]
                        logits = model.decoder.compute_logits(shard_h)
                        probs = torch.softmax(logits / self.args.sampling_temp, 2).clamp(min=1e-8, max=1.)
                        input_onehot = torch.nn.functional.one_hot(input_tokens[start:end], logits.shape[2])
                        if i == 0:
                            combined_probs = probs
                        else:
                            combined_probs = mask[start:end, :, None] * probs + (1 - mask[start:end, :, None]) * input_onehot
                        combined_probs = combined_probs.clamp(min=1e-8, max=1.)
                        out_tokens = combined_probs.argmax(2)
                        out_tokens_stack.append(out_tokens)
                    out_tokens = torch.cat(out_tokens_stack, 0)
                    out_tokens = out_tokens * decoder_padding_mask.logical_not() + y_input * decoder_padding_mask
            else:
                self.strategy.end_iteration = random_refine_step - 1
                out_tokens, _ = self.strategy.generate(model, encoder_out, y_input, self.tgt_dict)
        if was_training:
            model.train()
        return out_tokens

    def random_mask(self, encoder_out, y_input):
        baseline = self.task.baseline_model()
        input_mask = y_input.clone().float().fill_(0.1).bernoulli().bool()
        y_mask = torch.ne(y_input, 1)
        masked_input = (y_input * input_mask.logical_not() + 4 * input_mask) * y_mask + y_input * y_mask.logical_not()
        with torch.no_grad():
            logits, _ = baseline.decoder(masked_input, encoder_out=encoder_out)
            probs = torch.softmax(logits / self.args.sampling_temp, 2)
            B, T, _ = probs.shape
            y_sample = torch.multinomial(probs.view(B*T, -1), 1).view(B, T)
            y_sample = y_sample * y_mask + y_input * y_mask.logical_not()
        return y_sample, input_mask * y_mask, probs

    def compute_pnet_loss(self, model, encoder_out, y_input, sample, reduce=False):
        y_input = sample["net_input"]["prev_output_tokens"]
        y_mask = (y_input != 1)
        cmlm_mask = (y_input == 4)
        true_target = sample["true_target"]
        h, decoder_padding_mask, decoder_out = model.decoder.compute_pnet_firsthalf(y_input, encoder_out)
        pnet_out = decoder_out["pnet_out"]
        mask = torch.sigmoid(pnet_out)
        logits = model.decoder.compute_pnet_secondhalf(h, decoder_padding_mask, mask, encoder_out)
        probs = torch.softmax(logits / self.args.sampling_temp, 2).clamp(min=1e-8, max=1.)
        input_onehot = torch.nn.functional.one_hot(y_input, logits.shape[2])
        combined_probs = mask[:, :, None] * probs + (1 - mask[:, :, None]) * input_onehot
        combined_probs = combined_probs.clamp(min=1e-8, max=1.)
        B, T, _ = combined_probs.shape
        if self.args.offpolicy:
            y_sample, input_mask, proposal_probs = self.random_mask(encoder_out, y_input)
        else:
            y_sample = torch.multinomial(combined_probs.view(B*T, -1), 1).view(B, T)
        log_dist = torch.log(combined_probs)
        logp_mat = log_dist.view(B*T, -1)[torch.arange(B*T), y_sample.flatten()].view(B, T)
        y_pred = combined_probs.argmax(2)
        # Compute rewards
        next_absreward = self.compute_reward(y_pred * y_mask, true_target * y_mask)
        sample_absreward = self.compute_reward(y_sample * y_mask, true_target * y_mask)
        baseline_reward = self.compute_reward(y_input * y_mask, true_target * y_mask)
        relative_reward = sample_absreward - baseline_reward
        pred_relative_reweard = next_absreward - baseline_reward
        if self.args.tokenwise:
            change_mask = torch.ne(y_sample, y_input) * y_mask
            relative_reward = relative_reward / (change_mask.sum(1) + 1e-8)
            if self.args.offpolicy:
                with torch.no_grad():
                    imp = torch.exp(logp_mat) / proposal_probs.view(B*T, -1)[torch.arange(B*T), y_sample.flatten()].view(B,T) * y_mask
            else:
                imp = 1.
            loss = - (relative_reward[:, None] * imp * logp_mat * change_mask).sum(1) / (change_mask.sum(1) + 1e-8)
        else:
            logp = (logp_mat * y_mask).sum(1)
            loss = - relative_reward * logp
        if reduce:
            loss = loss.sum()
            relative_reward = relative_reward.sum()
            next_absreward = next_absreward.sum()
            pred_relative_reweard = pred_relative_reweard.sum()
        return loss, pred_relative_reweard, relative_reward, next_absreward


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
        if getattr(self.args, "pnet2", False) and sample["random_refine_step"] > 0:
            loss, reward, relative_reward, absreward = self.compute_pnet_loss(model, encoder_out, y_input, sample, reduce=reduce)

        else:
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

        if getattr(self.args, "pnet", False):
            # Training the confidence prediction network
            pnet_out = net_output[1]["pnet_out"]
            dist = torch.distributions.normal.Normal(pnet_out, 1.)
            if not model.training:
                sampled_p = pnet_out
            else:
                sampled_p = dist.sample()
            p_logp = dist.log_prob(sampled_p)
            t = sample["random_refine_step"]
            tokens_t = logits.argmax(2)
            tokens_t = tokens_t * y_mask + y_input * y_mask.logical_not()
            masked_tokens_t = tokens_t.clone()
            self.strategy.remask(masked_tokens_t, sampled_p, t + 1, self.args.refinetot)
            with torch.no_grad():
                model.decoder.select_decoder(t + 1)
                next_logits = model.decoder(masked_tokens_t, encoder_out=encoder_out)[0]
            next_tokens = next_logits.argmax(2)
            next_tokens = next_tokens * y_mask + y_input * y_mask.logical_not()
            next_absreward = self.compute_reward(next_tokens * y_mask, true_target * y_mask)
            baseline_reward = self.compute_reward(tokens_t * y_mask, true_target * y_mask)
            relative_reward = next_absreward - baseline_reward
            logp = (p_logp * y_mask).sum(1)
            loss = - relative_reward * logp
            if reduce:
                loss = loss.sum()
                relative_reward = relative_reward.sum()
                next_absreward = next_absreward.sum()
            return loss, relative_reward, relative_reward, next_absreward

        # >>>>>> Compute policy >>>>>>>>
        probs = torch.softmax(logits / self.args.sampling_temp, 2).clamp(min=1e-8, max=1.)
        if self.masker and model.decoder.selected_decoder > 0:
            input_onehot = torch.nn.functional.one_hot(y_input, logits.shape[2])
            sampled_mask = net_output[1]["sampled_mask"]
            probs = sampled_mask[:, :, None] * probs + (1 - sampled_mask[:, :, None]) * input_onehot
        probs = probs.clamp(min=1e-8, max=1.)
        log_probs = torch.log(probs)
        y_pred = probs.argmax(2)
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
        baseline_reward = self.compute_reward(baseline_ypred * y_mask, true_target * y_mask)

        # >>>>>> Obtain sample >>>>>>>>
        B, T, _ = probs.shape
        y_sample = torch.multinomial(probs.view(B*T, -1), 1).view(B, T)
        if not self.masker:
            y_sample = y_sample * cmlm_mask + y_input * cmlm_mask.logical_not()
        logp_mat = log_probs.view(B*T, -1)[torch.arange(B*T), y_sample.flatten()].view(B, T)
        # >>>>>> Compute reward >>>>>>>>
        sample_reward = self.compute_reward(y_sample * y_mask, true_target * y_mask)
        absreward = self.compute_reward(y_pred * y_mask, true_target * y_mask)
        reward = absreward - baseline_reward
        relative_reward = sample_reward - baseline_reward
        # >>>>>> Only for on-policy training >>>>>>>>
        if self.args.tokenwise:
            reward = reward / cmlm_mask.sum(1)
            relative_reward = relative_reward / cmlm_mask.sum(1)
            loss = (- relative_reward[:, None] * logp_mat * cmlm_mask).sum(1) / cmlm_mask.sum(1) # + self.args.klreg * kldiv / cmlm_mask.sum(1)
        else:
            if self.masker:
                if getattr(self.args, "masker_hard", False) and model.decoder.selected_decoder > 0:
                    sampled_mask = net_output[1]["sampled_mask"]
                    logp = (logp_mat * sampled_mask * y_mask).sum(1)
                else:
                    logp = (logp_mat * y_mask).sum(1)

            else:
                logp = (logp_mat * cmlm_mask).sum(1)
            loss = - relative_reward * logp # + self.args.klreg * kldiv
            if getattr(self.args, "masker_hard", False) and model.decoder.selected_decoder > 0:
                masking_prob = net_output[1]["masking_prob"]
                log_maskp_mat = torch.log(masking_prob)
                log_maskp = (log_maskp_mat * y_mask).sum(1)
                loss += - relative_reward * log_maskp

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