#!/usr/bin/env python3
# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,
#                                                       Zengwei Yao,
#                                                       Yifan Yang,
#                                                       Daniel Povey)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
HuBERT-style pre-training of the causal Zipformer2 encoder (the same medium
encoder as egs/ami/TSOT) on fbank features, with WavLM Base+ layer-9
k-means (500) labels. LibriSpeech 960 h + AMI IHM non-overlap train pieces.

Adapted from egs/ami/TSOT/zipformer/train.py (training infrastructure,
config files, checkpoint policy) and egs/librispeech/SSL/zipformer/pretrain.py
(k2SSL: schedule, Eden warmup, masked-prediction loss); changes:
  - model: ssl_model.FbankHubertModel (encoder_embed + encoder built exactly
    as for ASR, so they load unchanged into the t-SOT model)
  - data: ssl_datamodule.FbankHubertDataModule (crop to the shortest
    utterance of the batch, on-the-fly per-utterance fbank)
  - loss: CE on masked frames (sum); "frames" = number of masked frames
  - validation on ls_dev-other and ami_ihm_dev separately;
    best-valid-loss uses the two sets pooled. Validation masks are seeded
    per batch and the causal encoder uses a fixed chunk (640 ms by default),
    so validation losses are comparable across steps.
  - --bi-label (default false): also predict the secondary speaker's labels
    (SIL outside its region) with a second head; validation then also runs on
    mixed copies of the dev sets (<set>_mix, pooled as all_mix).

Usage:

./zipformer/pretrain.py --config conf/pretrain_zipformer_causal_medium.yaml
"""


import os

# Variable batch shapes fragment the CUDA caching allocator; expandable
# segments avoid this. Must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import optim
import torch

if TYPE_CHECKING:
    from torch.amp import GradScaler

import torch.multiprocessing as mp
import torch.nn as nn
from config_utils import parse_args_with_config, save_config
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from optim import Eden, ScaledAdam
from scaling import ScheduledFloat
from ssl_datamodule import FbankHubertDataModule
from ssl_model import FbankHubertModel
from subsampling import Conv2dSubsampling
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer2

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    create_grad_scaler,
    get_parameter_groups_with_lrs,
    setup_logger,
    str2bool,
    torch_autocast,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


class SslMetrics(MetricsTracker):
    """MetricsTracker whose printed / tensorboard values are divided by
    "frames", except "<name>_num" / "<name>_den" pairs, shown as one ratio
    "<name>" (e.g. accuracy on secondary-speech frames only)."""

    def __add__(self, other: MetricsTracker) -> "SslMetrics":
        ans = SslMetrics()
        ans.update(MetricsTracker.__add__(self, other))
        return ans

    def __mul__(self, alpha: float) -> "SslMetrics":
        ans = SslMetrics()
        ans.update(MetricsTracker.__mul__(self, alpha))
        return ans

    def norm_items(self):
        num_frames = self["frames"] if "frames" in self else 1
        ans = []
        for k, v in self.items():
            if k == "frames" or k.endswith("_den"):
                continue
            if k.endswith("_num"):
                den = self.get(k[:-4] + "_den", 0)
                ans.append((k[:-4], float(v) / den if den else float("nan")))
            else:
                ans.append((k, float(v) / num_frames))
        return ans


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def add_model_arguments(parser: argparse.ArgumentParser):
    # Encoder options: same names and defaults as egs/ami/TSOT/zipformer/train.py
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    # SSL options (defaults from k2SSL hubert_ce.py / fairseq HuBERT)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=500,
        help="Number of k-means clusters (prediction targets).",
    )

    parser.add_argument(
        "--label-shift",
        type=int,
        default=2,
        help="encoder_embed frame j predicts label j + label-shift (see ssl_model.py).",
    )

    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.8,
        help="Probability for each frame to be the start of a masked span.",
    )

    parser.add_argument(
        "--mask-length",
        type=int,
        default=10,
        help="Length (in 50 Hz frames) of each masked span.",
    )

    parser.add_argument(
        "--bi-label",
        type=str2bool,
        default=False,
        help="Bi-label training: also predict the labels of the secondary "
        "(mixed-in) utterance, SIL where there is none (see ssl_model.py).",
    )

    parser.add_argument(
        "--sec-loss-scale",
        type=float,
        default=1.0,
        help="Weight of the secondary-label loss (only with --bi-label).",
    )

    parser.add_argument(
        "--sec-spk-weight",
        type=float,
        default=1.0,
        help="Bi-label: weight of the secondary-label CE on frames with a "
        "secondary speaker (non-SIL; SIL frames have weight 1). 1: plain CE.",
    )

    parser.add_argument(
        "--logit-temp",
        type=float,
        default=0.1,
        help="Temperature dividing the logits.",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=300,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.045, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=10.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--warmup-batches",
        type=float,
        default=5000,
        help="Eden warmup steps",
    )

    parser.add_argument(
        "--warmup-start",
        type=float,
        default=0,
        help="Eden warmup start learning rate (relative)",
    )

    parser.add_argument(
        "--ref-duration",
        type=float,
        default=600,
        help="Reference batch duration for purposes of adjusting batch counts for setting various "
        "schedules inside the model",
    )

    parser.add_argument(
        "--min-keep-duration",
        type=float,
        default=1.0,
        help="Drop training utterances shorter than this (s). k2SSL uses 2.0 "
        "(min_keep_size 32000); all our data is >= 1 s.",
    )

    parser.add_argument(
        "--valid-chunk-size",
        type=int,
        default=32,
        help="Causal model: chunk size (50 Hz frames) used for all validation "
        "batches (training samples it per batch from --chunk-size). 32 = 640 ms.",
    )

    parser.add_argument(
        "--valid-left-context-frames",
        type=int,
        default=128,
        help="Causal model: left context (frames) used for validation.",
    )

    parser.add_argument(
        "--valid-interval",
        type=int,
        default=3000,
        help="Compute validation loss every this many batches.",
    )

    parser.add_argument(
        "--save-epoch-every",
        type=int,
        default=1,
        help="Keep epoch-N.pt only when N is a multiple of this (the latest "
        "epoch is always kept for resuming). 1 keeps every epoch (icefall default).",
    )

    parser.add_argument(
        "--save-best-checkpoints",
        type=str2bool,
        default=True,
        help="Copy the epoch checkpoint to best-train-loss.pt / best-valid-loss.pt.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--sanity-check",
        type=str2bool,
        default=True,
        help="Run the pessimistic-batch OOM check before training.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=4000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        0 disables these step-based checkpoints (resuming is then only
        possible from an epoch checkpoint).
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use bf16 in AMP.",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

        - best_train_loss / best_valid_loss: best masked CE per frame so far.
        - best_train_epoch / best_valid_epoch: the epochs where they occurred.
        - batch_idx_train: number of batches trained so far across epochs.
        - log_interval: print training loss if batch_idx % log_interval is 0
        - reset_interval: decay constant of the running training statistics
        - feature_dim: the model input dim (80-dim fbank).
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 2,  # encoder output kept at 50 Hz
            "env_info": get_env_info(),
        }
    )

    return params


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def get_encoder_embed(params: AttributeDict) -> nn.Module:
    # Same as egs/ami/TSOT/zipformer/train.py: (N, T, 80) -> (N, (T - 7) // 2, encoder_dim[0])
    encoder_embed = Conv2dSubsampling(
        in_channels=params.feature_dim,
        out_channels=_to_int_tuple(params.encoder_dim)[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    return encoder_embed


def get_encoder_model(params: AttributeDict) -> nn.Module:
    # Same as egs/ami/TSOT/zipformer/train.py; FbankHubertModel replaces the
    # final downsample_output by an identity (50 Hz output).
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def get_model(params: AttributeDict) -> nn.Module:
    model = FbankHubertModel(
        encoder_embed=get_encoder_embed(params),
        encoder=get_encoder_model(params),
        embed_dim=_to_int_tuple(params.encoder_dim)[0],
        encoder_out_dim=max(_to_int_tuple(params.encoder_dim)),
        num_classes=params.num_classes,
        label_shift=params.label_shift,
        mask_prob=params.mask_prob,
        mask_length=params.mask_length,
        logit_temp=params.logit_temp,
        bi_label=params.bi_label,
        sec_loss_scale=params.sec_loss_scale,
        sec_spk_weight=params.sec_spk_weight,
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional["GradScaler"] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file."""
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.save_best_checkpoints:
        if params.best_train_epoch == params.cur_epoch:
            best_train_filename = params.exp_dir / "best-train-loss.pt"
            copyfile(src=filename, dst=best_train_filename)

        if params.best_valid_epoch == params.cur_epoch:
            best_valid_filename = params.exp_dir / "best-valid-loss.pt"
            copyfile(src=filename, dst=best_valid_filename)

    # Keep only every `save_epoch_every`-th epoch checkpoint, plus the latest
    # one (needed to resume with --start-epoch).
    prev_epoch = params.cur_epoch - 1
    if (
        params.save_epoch_every > 1
        and prev_epoch >= 1
        and prev_epoch % params.save_epoch_every != 0
    ):
        prev_filename = params.exp_dir / f"epoch-{prev_epoch}.pt"
        if prev_filename.is_file():
            prev_filename.unlink()
            logging.info(f"Removed {prev_filename} (save_epoch_every={params.save_epoch_every})")


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    batch: dict,
    is_training: bool,
    mask_seed: Optional[int] = None,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute the masked-prediction loss (CE summed over masked frames).

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The FbankHubertModel (possibly wrapped in DDP).
      batch:
        A batch from ssl_datamodule.FbankHubertDataset.
      is_training:
        True for training. False for validation.
      mask_seed:
        If given, the masks are a deterministic function of it (validation).
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    features = batch["features"].to(device)
    feature_lens = batch["feature_lens"].to(device)
    labels = batch["labels"]

    with torch.set_grad_enabled(is_training):
        loss, stats = model(features, feature_lens, labels, mask=True, mask_seed=mask_seed,
                            sec_labels=batch.get("sec_labels"))

    assert loss.requires_grad == is_training

    info = SslMetrics()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = stats["frames"].item()  # masked frames with a label

    # Note: We use reduction=sum while computing the loss.
    # Printed values are divided by "frames": loss per frame, accuracy.
    info["loss"] = loss.detach().cpu().item()
    info["acc"] = stats["correct"].item()
    if params.bi_label:
        # loss = pri_loss + sec_loss_scale * sec_loss; sec_acc over all masked
        # frames; sec_spk_frac = share of frames with a secondary speaker;
        # sec_spk_acc / sec_sil_acc = accuracy on those / on the SIL frames.
        info["pri_loss"] = stats["pri_loss"].item()
        info["sec_loss"] = stats["sec_loss"].item()
        info["sec_acc"] = stats["sec_correct"].item()
        info["sec_spk_frac"] = stats["sec_spk_frames"].item()
        info["sec_spk_acc_num"] = stats["sec_spk_correct"].item()
        info["sec_spk_acc_den"] = stats["sec_spk_frames"].item()
        info["sec_sil_acc_num"] = stats["sec_sil_correct"].item()
        info["sec_sil_acc_den"] = stats["sec_sil_frames"].item()
    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    valid_dls: Dict[str, torch.utils.data.DataLoader],
    world_size: int = 1,
) -> Dict[str, MetricsTracker]:
    """Run validation on each dev set; returns {name: tracker, "all": pooled
    clean sets, "all_mix": pooled mixed sets (bi-label only)}.
    A causal encoder is evaluated with one fixed chunk size / left context
    (--valid-chunk-size / --valid-left-context-frames) instead of the random
    per-batch choice of training, so validations are comparable."""
    model.eval()
    encoder = (model.module if isinstance(model, DDP) else model).encoder
    train_chunk = (encoder.chunk_size, encoder.left_context_frames)
    if params.causal:
        encoder.chunk_size = (params.valid_chunk_size,)
        encoder.left_context_frames = (params.valid_left_context_frames,)

    results = {}
    tot_all = SslMetrics()
    tot_mix = SslMetrics()
    for set_idx, (name, dl) in enumerate(valid_dls.items()):
        tot_loss = SslMetrics()
        for batch_idx, batch in enumerate(dl):
            loss, loss_info = compute_loss(
                params=params,
                model=model,
                batch=batch,
                is_training=False,
                # same masks at every validation (and on every rank)
                mask_seed=params.seed * 1000003 + set_idx * 100003 + batch_idx,
            )
            assert loss.requires_grad is False
            tot_loss = tot_loss + loss_info

        if world_size > 1:
            tot_loss.reduce(loss.device)
        results[name] = tot_loss
        if name.endswith("_mix"):
            tot_mix = tot_mix + tot_loss
        else:
            tot_all = tot_all + tot_loss
    results["all"] = tot_all
    if "frames" in tot_mix:
        results["all_mix"] = tot_mix
    encoder.chunk_size, encoder.left_context_frames = train_chunk

    loss_value = tot_all["loss"] / tot_all["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return results


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    train_dl: torch.utils.data.DataLoader,
    valid_dls: Dict[str, torch.utils.data.DataLoader],
    scaler: "GradScaler",
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all masked frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.
    """
    model.train()

    tot_loss = SslMetrics()

    saved_bad_model = False

    def save_bad_model(suffix: str = ""):
        save_checkpoint_impl(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=0,
        )

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx % 10 == 0:
            set_batch_count(model, get_adjusted_batch_count(params))

        params.batch_idx_train += 1
        batch_size = len(batch["cuts"])

        try:
            with torch_autocast(enabled=params.use_autocast, dtype=params.dtype):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception: {e}.")
            save_bad_model()
            display_and_save_batch(batch, params=params)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.save_every_n > 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if params.use_autocast:
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                    if not params.inf_check:
                        register_inf_check_hooks(model)
                logging.warning(f"Grad scale is small: {cur_grad_scale}")

            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            if (
                batch_idx % 25 == 0
                and cur_grad_scale < 2.0
                or batch_idx % 100 == 0
                and cur_grad_scale < 8.0
                or batch_idx % 400 == 0
                and cur_grad_scale < 32.0
            ):
                scaler.update(cur_grad_scale * 2.0)

        if batch_idx % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_autocast else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"crop: {batch['features'].shape[1] / 100:.2f}s, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_autocast else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_autocast:
                    tb_writer.add_scalar(
                        "train/grad_scale", cur_grad_scale, params.batch_idx_train
                    )

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
            valid_infos = compute_validation_loss(
                params=params,
                model=model,
                valid_dls=valid_dls,
                world_size=world_size,
            )
            model.train()
            for name, info in valid_infos.items():
                logging.info(f"Epoch {params.cur_epoch}, validation {name}: {info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                for name, info in valid_infos.items():
                    info.write_summary(
                        tb_writer, f"train/valid_{name}_", params.batch_idx_train
                    )

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    if params.use_bf16:  # amp + bf16
        assert torch.cuda.is_bf16_supported(), "Your GPU does not support bf16!"
        assert not params.use_fp16, "You can only use either fp16 or bf16"
        params.dtype = torch.bfloat16
        params.use_autocast = True
    elif params.use_fp16:  # amp + fp16
        params.dtype = torch.float16
        params.use_autocast = True
    else:  # fp32
        params.dtype = torch.float32
        params.use_autocast = False

    logging.info(f"Using dtype={params.dtype}")
    logging.info(f"Use AMP={params.use_autocast}")

    logging.info(params)

    logging.info("About to create model")
    model = get_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")
    num_enc = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n.startswith(("encoder_embed.", "encoder."))
    )
    logging.info(f"Number of encoder_embed + encoder parameters: {num_enc}")

    assert params.save_every_n == 0 or params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(
        optimizer,
        params.lr_batches,
        params.lr_epochs,
        params.warmup_batches,
        params.warmup_start,
    )

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    data = FbankHubertDataModule(args)

    train_cuts = data.train_cuts()

    def remove_short_utt(c: Cut):
        # Long utterances are cropped to --max-sample-size in the dataset.
        return c.duration >= params.min_keep_duration

    train_cuts = train_cuts.filter(remove_short_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = data.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_dls = {
        name: data.valid_dataloaders(cuts) for name, cuts in data.dev_cuts().items()
    }
    if params.bi_label:
        valid_dls.update(
            {f"{name}_mix": data.valid_dataloaders(cuts, mixed=True)
             for name, cuts in data.dev_cuts().items()}
        )

    if params.sanity_check and not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            params=params,
        )

    scaler = create_grad_scaler(enabled=params.use_autocast, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dls=valid_dls,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
) -> None:
    """Display the batch statistics and save the batch into disk."""
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    logging.info(f"features shape: {batch['features'].shape}")
    logging.info(f"num labels: {sum(len(x) for x in batch['labels'])}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch_autocast(enabled=params.use_autocast, dtype=params.dtype):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    FbankHubertDataModule.add_arguments(parser)
    args = parse_args_with_config(parser)
    args.exp_dir = Path(args.exp_dir)
    saved = save_config(
        args, args.exp_dir, resume=args.start_epoch > 1 or args.start_batch > 0
    )
    print(f"Resolved config saved to {saved}")

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
