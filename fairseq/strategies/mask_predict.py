# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


@register_strategy('mask_predict')
class MaskPredict(DecodingStrategy):
    
    def __init__(self, args, exit_after_mask=False):
        super().__init__()
        self.iterations = args.decoding_iterations
        self.end_iteration = args.end_iteration
        self.exit_after_mask = exit_after_mask
        self.progressive = hasattr(args, "progressive") and args.progressive
    
    def generate(self, model, encoder_out, tgt_tokens, tgt_dict, return_token_probs=False):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)
        
        iterations = seq_len if self.iterations is None else self.iterations

        if self.progressive:
            model.decoder.select_decoder(0)
        tgt_tokens, token_probs = self.generate_non_autoregressive(model, encoder_out, tgt_tokens)
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        # print("Initialization: ", convert_tokens(tgt_dict, tgt_tokens[9]))

        for counter in range(1, iterations):
            if self.end_iteration != -1 and counter > self.end_iteration and not self.exit_after_mask:
                break
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())

            if self.end_iteration != -1 and counter > self.end_iteration and self.exit_after_mask:
                break
            # print("Step: ", counter+1)
            # print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[9]))
            if self.progressive:
                model.decoder.select_decoder(counter)
            decoder_out = model.decoder(tgt_tokens, encoder_out)
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)

            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)

            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            # print("Prediction: ", convert_tokens(tgt_dict, tgt_tokens[9]))
            if counter == self.end_iteration and not self.exit_after_mask:
                break
        
        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, token_probs if return_token_probs else lprobs
    
    def generate_non_autoregressive(self, model, encoder_out, tgt_tokens):
        decoder_out = model.decoder(tgt_tokens, encoder_out)
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)
