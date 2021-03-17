#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import sys
import os
import random
import numpy as np

import horovod.torch as hvd

import torch
from torch.nn import parallel
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
try:
    import nsml
    from nsml import DATASET_PATH
    HAS_NSML = True
except ImportError:
    HAS_NSML = False
    DATASET_PATH = "{}/works/data/CMLM".format(os.getenv("HOME"))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def distributed_init_hvd(args):
    import warnings
    import socket
    import torch.distributed as dist
    from fairseq.distributed_utils import suppress_output, is_master
    args.distributed_world_size = -1

    if torch.distributed.is_initialized():
        warnings.warn('Distributed is already initialized, cannot initialize twice!')
    else:
        print('| distributed init (rank {}): {}'.format(
            args.distributed_rank, args.distributed_init_method), flush=True)
        dist.init_process_group(
            backend="mpi",
            init_method=None,
            world_size=-1,
            rank=-1,
        )
        print('| initialized host {} as rank {}'.format(
            socket.gethostname(), args.distributed_rank), flush=True)

        dist.all_reduce(torch.rand(1))

        # suppress_output(is_master(args))

    args.distributed_rank = torch.distributed.get_rank()
    args.distributed_world_size = hvd.size()
    print("distributed", args.distributed_rank, args.distributed_world_size)
    return args.distributed_rank

def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_init_hvd(args)

    # Print args
    print(args)

    # if not HAS_NSML:
    #     args.data[0] = args.data[0].replace("/train", "")

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)

    if args.train_decoder_only:
        for name, param in model.named_parameters():
            if "decoder" not in name:
                param.requires_grad_(False)

    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Setup session
    if HAS_WANDB and distributed_utils.is_master(args):
        wandb.init(project="cmlm", config=args)
        wandb.watch(model)

    # Load pre-trained model
    data_token = args.data[0].split("/")[-1]
    if "bert" in args.arch:
        pretrained_path = "{}/train/pretrained_models/maskPredict_{}/checkpoint_best.pt".format(DATASET_PATH, data_token.split(".")[-1].replace("-", "_"))
        if not HAS_NSML:
            pretrained_path = pretrained_path.replace("/train", "")
        print("| loading", pretrained_path)
        state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_path)
        model.load_state_dict(state["model"], strict=True)
        baseline_model = task.build_model(args)
        baseline_model.load_state_dict(state["model"], strict=True)
        if torch.cuda.is_available():
            baseline_model.cuda()
        task.set_baseline_model(baseline_model)

    if not args.masking and HAS_NSML:
        def nsml_bind(model):
            def save(dir_path):
                state = {
                    'model': model.state_dict(),
                }
                torch.save(state, os.path.join(dir_path, 'best.pt'))

            def load(dir_path):
                state = torch.load(os.path.join(dir_path, 'best.pt'), map_location="cpu")
                model.load_state_dict(state['model'], strict=False)
                model.cuda()
                print('model loaded!')
            nsml.bind(save=save, load=load)
        nsml_bind(model)

    if args.load:
        print("loading model from session", args.load)
        if args.load.startswith("nsml://"):
            session = args.load.replace("nsml://", "")
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

    # Prepare for decoder wise training
    if args.decoder_wise_training:
        print("| Decoder wise training, start refinement step 0")
        progressive_training_step = 0
        assert args.ddp_backend == "c10d"
    else:
        progressive_training_step = None

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    if hasattr(args, "progressive") and args.progressive:
        for i in range(args.refinetot if not getattr(args, "pnet", False) else args.refinetot - 1):
            print("validating for refine step", i)
            validate(args, trainer, task, epoch_itr, valid_subsets, force_refine_step=i)
        print("---")
    validate(args, trainer, task, epoch_itr, valid_subsets)
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr, force_refine_step=progressive_training_step)
        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, force_refine_step=progressive_training_step)
        else:
            valid_losses = [None]

        if args.decoder_wise_training:
            progressive_training_step = update_num_to_refine_step(trainer.get_num_updates())

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            if HAS_NSML:
                if distributed_utils.is_master(args):
                    print("nsml save for epoch", epoch_itr.epoch)
                    nsml.save("epoch{}".format(epoch_itr.epoch))
            else:
                torch.save({"model": trainer.get_model().state_dict()}, "/tmp/epoch{}.pt".format(epoch_itr.epoch))
                if HAS_WANDB:
                    wandb.save("/tmp/epoch{}.pt".format(epoch_itr.epoch))
                # checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))

def update_num_to_refine_step(update_num):
    refine_step = int(update_num / 5000)
    return refine_step

def train(args, trainer, task, epoch_itr, force_refine_step=None):
    """Train the model for one epoch."""
    # Update parameters every N batches
    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    if hasattr(args, "progressive") and args.progressive:
        task.dataset("train").set_random_refine_step(args.refinetot, force_refine_step=force_refine_step)
    last_samples = None
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        if samples is None or len(samples) == 0:
            sys.stderr.write("Empty sample detected\n")
            sys.stderr.flush()
            samples = last_samples
        else:
            last_samples = samples
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue
        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, force_refine_step=force_refine_step)
            # if distributed_utils.is_master(args):
            #     print("saving:", trainer.get_num_updates())
            #     nsml.save(str(trainer.get_num_updates()))
            if not hasattr(checkpoint_utils.save_checkpoint, 'best') or is_better(valid_losses[0], checkpoint_utils.save_checkpoint.best):
                if distributed_utils.is_master(args):
                    print("saving checkpoint ...")
                    sys.stdout.flush()
                    if HAS_NSML:
                        nsml.save("best")
                    else:
                        torch.save({"model": trainer.get_model().state_dict()}, "/tmp/best.pt")
                    if HAS_WANDB:
                        wandb.save("/tmp/best.pt")
                    sys.stdout.flush()
                checkpoint_utils.save_checkpoint.best = valid_losses[0]

        if args.decoder_wise_training and update_num_to_refine_step(num_updates) != force_refine_step:
            if HAS_NSML:
                nsml.load("best")
            else:
                # Retrieve the model
                if distributed_utils.is_master(args):
                    state = torch.load("/tmp/best.pt", map_location="cpu")
                    trainer.model.load_state_dict(state["model"])
                # Sync
                assert isinstance(trainer.model, parallel.DistributedDataParallel)
                if isinstance(trainer.model, parallel.DistributedDataParallel):
                    trainer.model._sync_params()

            checkpoint_utils.save_checkpoint.best = 0.
            force_refine_step = update_num_to_refine_step(num_updates)
            trainer.criterion.pool.clear()
            print("| Start refinement step:", force_refine_step)


        if num_updates >= max_update:
            break

        if hasattr(args, "progressive") and args.progressive:
            task.dataset("train").set_random_refine_step(args.refinetot, force_refine_step=force_refine_step)

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets, force_refine_step=None):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_random = np.random.RandomState(3)
    valid_task_random = np.random.RandomState(3)
    if not hasattr(task, "random"):
        task.random = None
    task_random_bak = task.random
    task.random = valid_task_random
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        dataset = task.dataset(subset)
        if hasattr(dataset, "random"):
            random_bak = dataset.random
        else:
            random_bak = None
        dataset.random = valid_random
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        if hasattr(args, "progressive") and args.progressive:
            dataset.set_random_refine_step(args.refinetot, force_refine_step=force_refine_step)
        for sample in progress:
            if sample is None or len(sample) == 0:
                sys.stderr.write("empty valid sample detected\n")
                sys.stderr.flush()
                break
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

            if hasattr(args, "progressive") and args.progressive:
                dataset.set_random_refine_step(args.refinetot, force_refine_step=force_refine_step)
        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        valid_losses.append(stats[args.best_checkpoint_metric] if type(stats[args.best_checkpoint_metric]) == float else stats[args.best_checkpoint_metric].avg)
        dataset.random = random_bak

        if HAS_WANDB and distributed_utils.is_master(args):
            stat_dict = {}
            for k, v in stats.items():
                if isinstance(v, AverageMeter):
                    stat_dict[k] = v.val
                else:
                    stat_dict[k] = v
            wandb.log(stat_dict, step=trainer.get_num_updates())
    task.random = task_random_bak
    return valid_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        stats['best_loss'] = min(
            checkpoint_utils.save_checkpoint.best, stats['loss'].avg)
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
        main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--decoder-wise-training", action="store_true")
    parser.add_argument("--load", default="", type=str)
    parser.add_argument("--focus", default=-1, type=int)
    parser.add_argument("--masking", action="store_true")
    parser.add_argument("--train-decoder-only", action="store_true")
    parser.add_argument("--hvd", action="store_true")
    args = options.parse_args_and_arch(parser)

    assert args.hvd

    if getattr(args, "pnet", False) and args.load == "":
        print("training pnet requires loading a pretrained model")
        sys.exit()

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

        # distributed training
    if args.hvd:
        hvd.init()
        args.distributed_rank = hvd.rank()
        main(args, init_distributed=True)
    else:
        # single GPU training
        main(args)

    if args.mask:
        args.masking = True
        main(args)


if __name__ == '__main__':
    cli_main()
