# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import os
import sys
import torch
import numpy as np
import math
import torch.nn.functional as F
import re

from fairseq import pybleu, options, progress_bar, tasks, tokenizer, utils, strategies
from fairseq.meters import TimeMeter
from fairseq.data import IndexedCachedDataset
from fairseq.criterions.lib_sbleu import smoothed_bleu
from fairseq.strategies.strategy_utils import duplicate_encoder_out
from bleurt import score

PRETRAINED_PATH = ""

def main(args, checkpoint_name="best"):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'
    
    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)
    
    use_cuda = torch.cuda.is_available() and not args.cpu
    torch.manual_seed(args.seed)

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))
    args.taskobj = task

    # Set dictionaries
    #src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    dict = tgt_dict
    
    # Load decoding strategy
    strategy = strategies.setup_strategy(args)

    # Load ensemble
    if args.path.startswith("nsml://"):
        print("| loading nsml checkpoint", args.path)
        import nsml
        session = args.path.replace("nsml://", "")
        model = task.build_model(args)
        def load(dir_path):
            state = torch.load(os.path.join(dir_path, 'best.pt'))
            state_dict = state["model"]
            model.load_state_dict(state_dict)
            print("loaded")
        nsml.load(args.checkpoint_name, load_fn=load, session=session)
        models = [model.cuda()]
    elif args.path == "pretrain":
        from nsml import DATASET_PATH
        from fairseq import checkpoint_utils
        data_token = "en-de"
        pretrained_path = "{}/train/pretrained_models/maskPredict_{}/checkpoint_best.pt".format(DATASET_PATH, data_token.split(".")[-1].replace("-", "_"))
        print("| loading", pretrained_path)
        model = task.build_model(args)
        state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_path)
        model.load_state_dict(state["model"], strict=True)
        models = [model.cuda()]
    elif args.path.startswith("wb://"):
        print("| loading wb checkpoint", args.path)
        import wandb
        wandb.restore("best.pt", args.path.replace("wb://", ""), root="/tmp/")
        assert os.path.exists("/tmp/best.pt")
        state = torch.load("/tmp/best.pt")
        model = task.build_model(args)
        model.load_state_dict(state["model"])
        models = [model.cuda()]
    elif args.path.startswith("http://"):
        print("| loading http checkpoint", args.path)
        url = "http://trains.deeplearn.org:8081/{}".format(args.path.replace("http://", ""))
        os.system("curl -o /tmp/model.pt {}".format(url))
        state = torch.load("/tmp/model.pt")
        model = task.build_model(args)
        model.load_state_dict(state["model"])
        models = [model.cuda()]
    else:
        print('| loading model(s) from {}'.format(args.path))
        models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides))
        models = [model.cuda() for model in models]

    original_target_dataset = None
    assert args.original_target
    if args.original_target:
        original_target_dataset = IndexedCachedDataset(args.original_target, fix_lua_indexing=True)

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)
    
    results = []
    scorer = pybleu.PyBleuScorer()
    num_sentences = 0
    has_target = True
    timer = TimeMeter()
    rel_reward_log = []

    with progress_bar.build_progress_bar(args, itr) as t:

        translations = generate_batched_itr(t, strategy, models, tgt_dict, length_beam_size=args.length_beam, use_gold_target_len=args.gold_target_len)
        for sample_id, src_tokens, target_tokens, hypos, logp in translations:

            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None

            # Either retrieve the original sentences or regenerate them from tokens.
            distill_str = dict.string(target_tokens, args.remove_bpe, escape_unk=True)
            hypo_str = dict.string(hypos, args.remove_bpe, escape_unk=True)
            hypo_str_bpe = dict.string(hypos, None, escape_unk=True)

            # Compute reward
            original_target_dataset.prefetch([sample_id])
            orig_target = dict.string(original_target_dataset[sample_id], args.remove_bpe, escape_unk=True)
            hypo_reward = smoothed_bleu(hypo_str.split(), orig_target.split())
            distill_reward = smoothed_bleu(distill_str.split(), orig_target.split())
            rel_reward = hypo_reward - distill_reward
            rel_reward_log.append(rel_reward)

            print("{} | {:.4f} | {:.4f} | {}".format(sample_id, rel_reward, logp, hypo_str_bpe))
    print("mean rel reward:", np.mean(rel_reward_log))

def dehyphenate(sent):
    return re.sub(r'(\S)-(\S)', r'\1 ##AT##-##AT## \2', sent).replace('##AT##', '@')


def generate_batched_itr(data_itr, strategy, models, tgt_dict, length_beam_size=None, use_gold_target_len=False, cuda=True):
    """Iterate over a batched dataset and yield individual translations.
     Args:
        maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
    """
    for sample in data_itr:
        s = utils.move_to_cuda(sample) if cuda else sample
        if 'net_input' not in s:
            continue
        input = s['net_input']

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in input.items()
            if k != 'prev_output_tokens'
        }

        with torch.no_grad():
            gold_target_len = s['target'].ne(tgt_dict.pad()).sum(-1) if use_gold_target_len else None
            hypos, seq_logp = generate(strategy, encoder_input, models, tgt_dict, length_beam_size, gold_target_len, target_seq=sample["target"])
            for batch in range(hypos.size(0)):
                src = utils.strip_pad(input['src_tokens'][batch].data, tgt_dict.pad())
                ref = utils.strip_pad(s['target'][batch].data, tgt_dict.pad()) if s['target'] is not None else None
                hypo = utils.strip_pad(hypos[batch], tgt_dict.pad())
                example_id = s['id'][batch].data
                yield example_id, src, ref, hypo, seq_logp[batch].data


def generate(strategy, encoder_input, models, tgt_dict, length_beam_size, gold_target_len, target_seq=None):
    assert len(models) == 1
    model = models[0]

    src_tokens = encoder_input['src_tokens']
    src_tokens = src_tokens.new(src_tokens.tolist())
    bsz = src_tokens.size(0)
    
    encoder_out = model.encoder(**encoder_input)

    target_seq = target_seq.cuda()
    pad_mask  = target_seq.eq(1)
    rand_mask = target_seq.clone().float().fill_(0.5).bernoulli().bool()
    input_seq = target_seq * rand_mask.logical_not() + target_seq.clone().fill_(4) * rand_mask
    input_seq = pad_mask * target_seq + pad_mask.logical_not() * input_seq

    logits, _ = model.decoder(input_seq, encoder_out)

    probs = torch.softmax(logits * 2, dim=2)
    B, T, _ = probs.shape
    sampled_tokens = probs.view(B*T, -1).multinomial(1).view(B, T)
    sampled_probs = probs.view(B * T, -1)[torch.arange(B*T), sampled_tokens.flatten()].view(B, T)
    sampled_logp = torch.log(sampled_probs)
    sampled_tokens = pad_mask * target_seq + pad_mask.logical_not() * sampled_tokens
    sampled_tokens = rand_mask * sampled_tokens + rand_mask.logical_not() * target_seq
    seq_logp = (sampled_logp * rand_mask * pad_mask.logical_not()).sum(1)

    return sampled_tokens, seq_logp


def predict_length_beam(gold_target_len, predicted_lengths, length_beam_size):
    if gold_target_len is not None:
        beam_starts = gold_target_len - (length_beam_size - 1) // 2
        beam_ends = gold_target_len + length_beam_size // 2 + 1
        beam = torch.stack([torch.arange(beam_starts[batch], beam_ends[batch], device=beam_starts.device) for batch in range(gold_target_len.size(0))], dim=0)
    else:
        beam = predicted_lengths.topk(length_beam_size, dim=1)[1]
    beam[beam < 2] = 2
    return beam


if __name__ == '__main__':
    parser = options.get_generation_parser()
    options.add_model_args(parser)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--stepwise", action="store_true")
    parser.add_argument("--semiat", action="store_true")
    parser.add_argument("--scan-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-name", type=str, default="best")
    parser.add_argument("--original-target", type=str, default="")
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--end-iteration", default=-1, type=int)
    args = options.parse_args_and_arch(parser)
    if args.all:
        for i in [0, 2, 4, 8, 10]:
            print("testing with iterations", i)
            args.decoding_iterations = i
            main(args)
    elif args.stepwise:
        for i in range(args.decoding_iterations):
            args.end_iteration = i
            print("testing until iteration {}".format(i))
            main(args)
    elif args.scan_checkpoints:
        for i in range(1, 31):
            update_id = i * 20
            print("Scan for update num:", update_id)
            main(args, checkpoint_name=str(update_id))
    else:
        main(args)
