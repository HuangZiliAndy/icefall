#!/usr/bin/env python3
"""
Save the model that zipformer/decode.py evaluates with
--epoch E --avg A --use-averaged-model 1 as a standalone checkpoint
(model weights only, no optimizer / sampler state).

The average is computed from the running averages (model_avg) stored in
epoch-{E-A}.pt and epoch-{E}.pt with
icefall.checkpoint.average_checkpoints_with_averaged_model, as in decode.py.

Usage:
  python local/export_averaged_model.py --exp-dir exp/cfg35 --epoch 50 --avg 10
  -> exp/cfg35/pretrained.pt  ({"model": state_dict, "epoch", "avg"})

Decode it with:
  ln -sf pretrained.pt exp/cfg35/epoch-999.pt
  ./decode.sh --epoch 999 --avg 1 --use-averaged-model 0 35 0
"""
import argparse
import logging
from pathlib import Path

import torch

from icefall.checkpoint import average_checkpoints_with_averaged_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=Path, required=True)
    p.add_argument("--epoch", type=int, default=50)
    p.add_argument("--avg", type=int, default=10)
    p.add_argument("--out", type=Path, default=None, help="default: <exp-dir>/pretrained.pt")
    return p.parse_args()


def main():
    args = get_args()
    start = args.exp_dir / f"epoch-{args.epoch - args.avg}.pt"
    end = args.exp_dir / f"epoch-{args.epoch}.pt"
    logging.info(f"averaging {start} (excluded) .. {end}")
    model = average_checkpoints_with_averaged_model(
        filename_start=str(start), filename_end=str(end), device=torch.device("cpu")
    )
    out = args.out or args.exp_dir / "pretrained.pt"
    torch.save({"model": model, "epoch": args.epoch, "avg": args.avg}, out)
    n = sum(v.numel() for v in model.values())
    logging.info(f"saved {out}: {len(model)} tensors, {n} values")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    main()
