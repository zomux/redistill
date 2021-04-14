#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch
import numpy as np
import os
import sys

from fairseq import pybleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.criterions.lib_sbleu import smoothed_bleu
from bleurt import score
from transformers import cached_path

try:
    import nsml
    HAS_NSML = True
except ImportError:
    HAS_NSML = False

def compute_reward(hypo_str, target_str):
    return smoothed_bleu(hypo_str.split(), target_str.split())

def main(args):
    global score
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    if args.reward == "bleurt" or args.eval_bleurt:
        sys.argv = sys.argv[:1]
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        bleurt_scorer = score.BleurtScorer(os.path.join(
            cached_path(
                "https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip",
                extract_compressed_file=True
            ), "bleurt-base-128"
        ))

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    if args.path.startswith("nsml://"):
        # NSML
        session = args.path.replace("nsml://", "")
        model = task.build_model(args)
        if ".pt" in session:
            session = session.replace(".pt", "")
            session, checkpoint_name = session.rsplit("/", 1)
        else:
            checkpoint_name = "best"
        if "-" in checkpoint_name:
            start, end = checkpoint_name.replace("epoch", "").split("-")
            checkpoints = ["epoch{}".format(i) for i in range(int(start), int(end) + 1)]
            print("| checkpoint average:", checkpoints)
            state_dict = None
            def load(dir_path):
                nonlocal state_dict, checkpoints
                state = torch.load(os.path.join(dir_path, 'best.pt'))
                model_state = state["model"]
                for k in model_state:
                    model_state[k] = model_state[k] / float(len(checkpoints))
                if state_dict is None:
                    state_dict = model_state
                else:
                    for k in state_dict:
                        state_dict[k] += model_state[k]
                print("checkpoint loaded")
            for checkpoint_name in checkpoints:
                nsml.load(checkpoint_name, load_fn=load, session=session)
            model.load_state_dict(state_dict)
        else:
            def load(dir_path):
                state = torch.load(os.path.join(dir_path, 'best.pt'))
                state_dict = state["model"]
                model.load_state_dict(state_dict)
                print("loaded")
            nsml.load(checkpoint_name, load_fn=load, session=session)
        models = [model.cuda()]
    elif "-" in args.path:
        model = task.build_model(args)
        print("loading model from", args.path)
        state_dict = None
        dir_path = os.path.dirname(args.path)
        fn = os.path.basename(args.path)
        if "-" in fn:
            start, end = fn.replace("epoch", "").replace(".pt", "").split("-")
            checkpoint_fns = ["epoch{}.pt".format(i) for i in range(int(start), int(end) + 1)]
        else:
            checkpoint_fns = [fn]
        for fn in checkpoint_fns:
            state = torch.load(os.path.join(dir_path, fn))
            model_state = state["model"]
            for k in model_state:
                model_state[k] = model_state[k] / float(len(checkpoint_fns))
            if state_dict is None:
                state_dict = model_state
            else:
                for k in state_dict:
                    state_dict[k] += model_state[k]
            print("checkpoint loaded")
        model.load_state_dict(state_dict)
        models = [model.cuda()]
    else:
        model = task.build_model(args)
        state = torch.load(args.path)
        model_state = state["model"]
        model.load_state_dict(model_state)
        models = [model.cuda()]

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

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
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    # Generate and compute BLEU score
    # if args.sacrebleu:
    #     scorer = bleu.SacrebleuScorer()
    # else:
    #     scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    scorer = pybleu.PyBleuScorer()
    num_sentences = 0
    has_target = True
    results = []
    best_rank_list = []
    if args.save_path:
        outf = open(args.save_path, "w")
    total_n = 0
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            hypo_target_pairs = []
            for i, sample_id in enumerate(sample['id'].tolist()):
                total_n += 1
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                if args.reward_sample or args.reward_check:
                    # Get sample
                    hypo_strs = []
                    rewards = []
                    for j, hypo in enumerate(hypos[i]):
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=None,
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=None,
                        )
                        hypo_strs.append(hypo_str)
                    if args.reward == "sbleu":
                        for hypo_str in hypo_strs:
                            hypo_str_nobpe = hypo_str.replace("@@ ", "")
                            rewards.append(compute_reward(hypo_str_nobpe, target_str))
                        best_idx = np.array(rewards).argmax()
                        if args.reward_check:
                            best_rank_list.append(best_idx)
                        if args.save_path:
                            if args.output_all:
                                for hypo_i in range(len(hypo_strs)):
                                    outf.write("{} | {:.4f} | {}\n".format(sample_id, rewards[hypo_i], hypo_strs[hypo_i]))
                            else:
                                outf.write("{} | {}\n".format(sample_id, hypo_strs[best_idx]))
                        else:
                            if args.output_all:
                                for hypo_i in range(len(hypo_strs)):
                                    print("{} | {:.4f} | {}".format(sample_id, rewards[hypo_i], hypo_strs[hypo_i]))
                            else:
                                print("{} | {}".format(sample_id, hypo_strs[best_idx]))
                            sys.stdout.flush()
                    elif args.reward == "bleurt":
                        hypo_target_pairs.append((
                            sample_id, target_str, hypo_strs
                        ))
                else:
                    # Normal translation
                    # Process top predictions
                    for j, hypo in enumerate(hypos[i][:args.nbest]):
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
                            remove_bpe=args.remove_bpe,
                        )

                        if not args.quiet:
                            print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                            print('P-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(
                                    lambda x: '{:.4f}'.format(x),
                                    hypo['positional_scores'].tolist(),
                                ))
                            ))

                            if args.print_alignment:
                                print('A-{}\t{}'.format(
                                    sample_id,
                                    ' '.join(map(lambda x: str(utils.item(x)), alignment))
                                ))

                        # Score only the top hypothesis
                        results.append((sample_id, target_str, hypo_str, float(hypo["positional_scores"].mean())))
                        if has_target and j == 0 and not args.reward_sample:
                            pass
                            # if align_dict is not None or args.remove_bpe is not None:
                                # Convert back to tokens for evaluation with unk replacement and/or without BPE
                                # target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                            # if args.save_path:
                            #     outf.write("{} | {}\n".format(sample_id, hypo_str))
                            # if j == 0 and not args.no_eval:
                            #     results.append((sample_id, target_str, hypo_str))
                            # if hasattr(scorer, 'add_string'):
                            #     scorer.add_string(target_str, hypo_str)
                            # else:
                            #     scorer.add(target_tokens, hypo_tokens)
            if args.save_amount > 0 and total_n > args.save_amount:
                break
            if args.reward_sample and bool(hypo_target_pairs):
                hypo_batch = []
                target_batch = []
                for _, target, hypo_strs in hypo_target_pairs:
                    hypo_batch.extend([h.replace("@@ ", "") for h in hypo_strs])
                    target_batch.extend([target_str] * len(hypo_strs))
                rewards = np.array(bleurt_scorer.score(target_batch, hypo_batch))
                base_i = 0
                for sample_id, _, hypo_strs in hypo_target_pairs:
                    start = base_i
                    end = base_i + len(hypo_strs)
                    best_idx = rewards[start:end].argmax()
                    if args.save_path:
                        if args.output_all:
                            for idx in range(start, end):
                                outf.write("{} | {:.4f} | {}\n".format(sample_id, float(rewards[idx]), hypo_strs[idx - start]))
                        else:
                            outf.write("{} | {}\n".format(sample_id, hypo_strs[best_idx]))
                    else:
                        if args.output_all:
                            for idx in range(start, end):
                                print("{} | {:.4f} | {}".format(sample_id, float(rewards[idx]), hypo_strs[idx - start]))
                        else:
                            print("{} | {}".format(sample_id, hypo_strs[best_idx]))
                        sys.stdout.flush()
                    base_i += len(hypo_strs)
            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if args.save_path and not args.reward_check and not args.reward_sample:
        results.sort()
        for sample_id, tgt, hyp, score in results:
            outf.write("{}\t{}\t{}\n".format(sample_id, score, hyp))
        print("results saved to", args.save_path)

    if args.reward_check:
        print("avg ranking of the best sample:", np.array(best_rank_list).mean())
        print("ratio of best sample ranked in the top:", (np.array(best_rank_list) == 0).mean())
    if has_target and not args.reward_sample and not args.reward_check and not args.no_eval:
        _, ref, out, _ = zip(*results)
        from fairseq.criterions.lib_sbleu import smoothed_bleu
        sbleu = np.mean([smoothed_bleu(p[1].split(), p[2].split()) for p in results])
        print("| SBLEU = {:.2f}".format(sbleu))
        if args.eval_bleurt:
            bleurt_scores = bleurt_scorer.score(references=[p[1] for p in results], candidates=[p[2] for p in results])
            print("| BLEURT = {:.4f}".format(np.mean((np.array(bleurt_scores)))))
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.score(ref, out)))
    return scorer


def cli_main():
    parser = options.get_generation_parser()
    options.add_model_args(parser)
    parser.add_argument("--reward-sample", action="store_true")
    parser.add_argument("--reward", default="sbleu")
    parser.add_argument("--reward-check", action="store_true")
    parser.add_argument("--output-all", action="store_true")
    parser.add_argument("--eval-bleurt", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--save-amount", type=int, default=-1)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
