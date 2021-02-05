# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import (
    FairseqDecoder, FairseqEncoder, FairseqLanguageModel,
    register_model, register_model_architecture,
    FairseqIncrementalDecoder, FairseqModel
)

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveSoftmax, CharacterTokenEmbedder, MultiheadAttention,
    SimpleSinusoidalPositionalEmbedding, LearnedPositionalEmbedding
)

from .bert_seq2seq import BertLayerNorm, TransformerEncoder, PositionalEmbedding, TransformerDecoderLayer, SelfTransformerDecoder

@register_model('bert_transformer_seq2seq')
class Transformer_nonautoregressive(FairseqModel):
    def __init__(self, encoder, decoder, progressive=False, light=False, masker=False):
        self.progressive = progressive
        self.light = light
        self.masker = True
        super().__init__(encoder, decoder)
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.beta.data.zero_()
            module.gamma.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-enc-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--no-dec-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--embedding-only', default=False, action='store_true',
                            help='if set, replaces the encoder with just token embeddings (could be complex e.g. bilm')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--bilm-model-dropout', default=0.1, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the model dropout for bilm')
        parser.add_argument('--bilm-attention-dropout', default=0.0, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the attention dropout for bilm')
        parser.add_argument('--bilm-relu-dropout', default=0.0, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the relu dropout for bilm')
        parser.add_argument('--bilm-mask-last-state', action='store_true',
                            help='if set, masks last state in bilm as is done during training')
        parser.add_argument('--bilm-add-bos', action='store_true',
                            help='if set, adds bos to input')
        parser.add_argument('--decoder-embed-scale', type=float,
                            help='scaling factor for embeddings used in decoder')
        parser.add_argument('--encoder-embed-scale', type=float,
                            help='scaling factor for embeddings used in encoder')
        parser.add_argument("--progressive", action="store_true")
        parser.add_argument("--masker", action="store_true", help="token masking model")
        parser.add_argument("--masker-light", action="store_true", help="token masking model")
        parser.add_argument("--masker-hard", action="store_true", help="token masking model")
        parser.add_argument("--light", action="store_true", help="train only the last decoder layer")
        parser.add_argument("--pnet", action="store_true", help="pnet")
        parser.add_argument("--refinestep", default=0, type=int)
        parser.add_argument("--refinetot", default=0, type=int)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        #for ds in task.datasets.values():
        #    ds.target_is_source = True

        # make sure all arguments are present in older models
        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, is_encoder, path=None):

            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, is_encoder=True, path=args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, is_encoder=True, path=args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, is_encoder=False, path=args.decoder_embed_path
            )

        light = hasattr(args, "light") and args.light
        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, args.encoder_embed_scale)
        masker = hasattr(args, "masker") and args.masker
        if hasattr(args, "progressive") and args.progressive:
            decoder = ProgressiveSelfTransformerDecoder(args, tgt_dict, decoder_embed_tokens, args.decoder_embed_scale, light=light, masker=masker)
        else:
            decoder = SelfTransformerDecoder(args, tgt_dict, decoder_embed_tokens, args.decoder_embed_scale)
        progressive = hasattr(args, "progressive") and args.progressive
        return Transformer_nonautoregressive(encoder, decoder, progressive=progressive, light=light, masker=masker)

    def load_state_dict(self, state_dict, strict=True):
        if self.masker:
            strict = False
        if self.progressive:
            keys = list(state_dict.keys())
            for key in keys:
                if self.light:
                    prefix = "decoder.layers.{}.".format(len(self.decoder.layers) - 1)
                    if key.startswith(prefix):
                        for i in range(len(self.decoder.last_decoder_layers)):
                            new_key = key.replace(prefix, "decoder.last_decoder_layers.{}.".format(i))
                            if new_key not in keys:
                                state_dict[new_key] = state_dict[key]
                else:
                    if key.startswith("decoder.layers"):
                        for i in range(len(self.decoder.layer_stack)):
                            new_key = key.replace("decoder.layers", "decoder.layer_stack.{}".format(i))
                            if new_key not in keys:
                                state_dict[new_key] = state_dict[key]
                        del state_dict[key]
        super().load_state_dict(state_dict, strict=strict)

class ProgressiveSelfTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, embed_tokens, embed_scale=None, no_encoder_attn=False, left_pad=False,
                 final_norm=True, light=False, masker=False):
        super().__init__(dictionary)
        self.args = args
        self.light = light
        self.masker = masker
        self.pnet = getattr(args, "pnet", False)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        self.embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(self.embed_dim) if embed_scale is None else embed_scale

        self.project_in_dim = nn.Linear(input_embed_dim, self.embed_dim,
                                     bias=False) if self.embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, self.embed_dim, self.padding_idx,
            #learned=args.decoder_learned_pos,
        ) if not args.no_dec_token_positional_embeddings else None

        if hasattr(args, "decoding_iterations") and args.decoding_iterations > 0:
            args.refinetot = args.decoding_iterations

        self.selected_decoder = 0
        if self.light:
            self.layers = nn.ModuleList()
            self.layers.extend([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ])
            # self.layers.requires_grad_(False)

            self.last_decoder_layers = nn.ModuleList()
            self.last_decoder_layers.extend([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.refinetot)
            ])
        else:
            self.layers = None  # switchable
            self.layer_stack = nn.ModuleList()
            for _ in range(args.refinetot):
                layers = nn.ModuleList([])
                layers.extend([
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(args.decoder_layers)
                ])
                self.layer_stack.append(layers)

        if self.pnet:
            self.pnet_layers = nn.ModuleList([
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(3)
                ])
            self.pnet_pred = nn.Linear(self.embed_dim, 1)

        if self.masker:
            self.masker_stack = nn.ModuleList()
            if hasattr(self.args, "masker_light") and self.args.masker_light:
                masker_layers = 1
            else:
                masker_layers = args.refinetot - 1
            for _ in range(masker_layers):
                layers = nn.ModuleList([])
                layers.extend([
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(3)
                ])
                self.masker_stack.append(layers)
            self.masker_predict_stack = nn.ModuleList()
            for _ in range(masker_layers):
                self.masker_predict_stack.append(nn.Linear(args.decoder_output_dim, 1))

        if getattr(self.args, "fitbase", False):
            self.baseline_nn = nn.Sequential(nn.Linear(self.embed_dim, int(self.embed_dim / 2)),
                                             nn.ReLU(),
                                             nn.Linear(int(self.embed_dim / 2), 1))

        self.adaptive_softmax = None

        self.project_out_dim = nn.Linear(self.embed_dim, output_embed_dim, bias=False) \
            if self.embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        self.load_softmax = not getattr(args, 'remove_head', False)

        if self.load_softmax:
            if args.adaptive_softmax_cutoff is not None:
                self.adaptive_softmax = AdaptiveSoftmax(
                    len(dictionary),
                    output_embed_dim,
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                    dropout=args.adaptive_softmax_dropout,
                    adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                    factor=args.adaptive_softmax_factor,
                    tie_proj=args.tie_adaptive_proj,
                )
            elif not self.share_input_output_embed:
                self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
                #nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = BertLayerNorm(self.embed_dim)
        if not self.share_input_output_embed:
            self.embed_out.requires_grad_(False)
        self.embed_tokens.requires_grad_(False)
        if self.pnet:
            for name, param in self.named_parameters():
                if "pnet" not in name:
                    param.requires_grad_(False)

    def select_decoder(self, id):
        self.selected_decoder = id

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, compute_logits=True):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        incremental_state=None
        
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
        ) if self.embed_positions is not None else None

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        # Masking token embeddings using masker
        if self.masker and self.selected_decoder > 0:
            masker_h = F.dropout(x, p=self.dropout, training=self.training).transpose(0, 1)
            if hasattr(self.args, "masker_light") and self.args.masker_light:
                masker_index = 0
            else:
                masker_index = self.selected_decoder - 1
            masker_layers = self.masker_stack[masker_index]
            for layer in masker_layers:
                masker_h, _ = layer(
                    masker_h,
                    encoder_out['encoder_out'] if encoder_out is not None else None,
                    encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                    decoder_padding_mask,
                )
            masker_h = masker_h.transpose(0, 1)
            masking_prob = torch.sigmoid(self.masker_predict_stack[masker_index](masker_h).sum(2))
            if getattr(self.args, "masker_hard", False):
                if self.training:
                    sampled_mask = torch.bernoulli(masking_prob).clamp(min=1e-8, max=1.0)
                else:
                    sampled_mask = torch.gt(masking_prob, 0.5) * 1.0
            else:
                sampled_mask = masking_prob
            # if not self.training:
            #     from fairseq.strategies.strategy_utils import assign_single_value_long, assign_single_value_byte
            #     assign_single_value_byte(masking_prob, decoder_padding_mask, 0.0)
            #     num_mask = (decoder_padding_mask.logical_not().sum(1) * (1.0 - self.selected_decoder / self.args.decoding_iterations)).long()
            #     mask_ind = self.select_worst(-masking_prob, num_mask)
            #     new_masking_prob = masking_prob * 0.
            #     assign_single_value_long(new_masking_prob, mask_ind, 1.0)
            #     masking_prob = new_masking_prob
            y_mask = prev_output_tokens != 1
            full_mask_y = (prev_output_tokens * 0 + 4) * y_mask + prev_output_tokens * y_mask.logical_not()
            mask_embed = self.embed_tokens(full_mask_y)
            x = sampled_mask[:, :, None] * mask_embed + (1 - sampled_mask)[:, :, None] * x

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        if self.light:
            decoder_layers = list(self.layers[:-1]) + [self.last_decoder_layers[self.selected_decoder]]
        else:
            decoder_layers = self.layer_stack[self.selected_decoder]
        # decoder layers
        for layer in decoder_layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                decoder_padding_mask,
            )
            inner_states.append(x)
        if self.normalize:
            x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        # print("xh", x[-1])

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        last_h = x
        if compute_logits:
            x = self.compute_logits(x)

        decoder_out_dict = {'attn': attn, 'inner_states': inner_states, 'predicted_lengths': encoder_out['predicted_lengths']}
        if self.masker and self.selected_decoder > 0:
            decoder_out_dict["masking_prob"] = masking_prob
            decoder_out_dict["sampled_mask"] = sampled_mask
        if self.pnet:
            h = last_h.transpose(0, 1)
            for layer in self.pnet_layers:
                h, attn = layer(
                    h,
                    encoder_out['encoder_out'] if encoder_out is not None else None,
                    encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                    decoder_padding_mask,
                )
            pnet_h = h.transpose(0, 1)
            pnet_out = self.pnet_pred(pnet_h)[:, :, 0]
            decoder_out_dict["pnet_out"] = pnet_out

        return x, decoder_out_dict

    def compute_logits(self, x):
        if self.adaptive_softmax is None and self.load_softmax:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)
        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        #return min(self.max_target_positions, self.embed_positions.max_positions())
        return self.max_target_positions
    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        pass


@register_model_architecture('bert_transformer_seq2seq', 'bert_transformer_seq2seq')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', args.encoder_embed_dim * 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', args.encoder_embed_dim // 64)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', args.encoder_attention_heads)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_enc_token_positional_embeddings = getattr(args, 'no_enc_token_positional_embeddings', False)
    args.no_dec_token_positional_embeddings = getattr(args, 'no_dec_token_positional_embeddings', False)
    args.embedding_only = getattr(args, 'embedding_only', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.decoder_embed_scale = getattr(args, 'decoder_embed_scale', None)
    args.encoder_embed_scale = getattr(args, 'encoder_embed_scale', None)

    args.bilm_mask_last_state = getattr(args, 'bilm_mask_last_state', False)
    args.bilm_add_bos = getattr(args, 'bilm_add_bos', False)



@register_model_architecture('bert_transformer_seq2seq', 'bert_transformer_seq2seq_big')
def bi_transformer_lm_big(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_architecture(args)