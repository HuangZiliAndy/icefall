#!/usr/bin/env python3
"""
Save the averaged pre-trained model over epochs (epoch - avg, epoch] as a
standalone checkpoint, for initialising the t-SOT encoder.

The average is computed from the running averages (model_avg) stored in
epoch-{epoch-avg}.pt and epoch-{epoch}.pt, as decode.py does with
--use-averaged-model 1 (icefall.checkpoint.average_checkpoints_with_averaged_model):
it is the uniform average of the snapshots taken every --average-period batches
within those epochs.

The model is rebuilt from <exp_dir>/config.yaml (the resolved training
options) and the averaged state dict is loaded with strict=True.

Usage:
  python zipformer/export_averaged_model.py --exp-dir exp/cfg3 --epoch 100 --avg 10
  -> exp/cfg3/epoch-100-avg-10.pt  ({"model": state_dict, "epoch", "avg", ...})
"""
import argparse
import logging
from pathlib import Path

import torch
import yaml

import pretrain
from icefall.checkpoint import average_checkpoints_with_averaged_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", type=Path, required=True)
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--avg", type=int, required=True)
    args = p.parse_args()

    # Rebuild the model exactly as in training.
    parser = pretrain.get_parser()
    pretrain.FbankHubertDataModule.add_arguments(parser)
    with open(args.exp_dir / "config.yaml") as f:
        cfg = {k.replace("-", "_"): v for k, v in yaml.safe_load(f).items()}
    known = {a.dest for a in parser._actions}
    parser.set_defaults(**{k: v for k, v in cfg.items() if k in known})
    train_args = parser.parse_args([])
    params = pretrain.get_params()
    params.update(vars(train_args))
    model = pretrain.get_model(params)

    start = args.exp_dir / f"epoch-{args.epoch - args.avg}.pt"
    end = args.exp_dir / f"epoch-{args.epoch}.pt"
    logging.info(f"Averaging over ({start.name}, {end.name}]")
    avg = average_checkpoints_with_averaged_model(filename_start=str(start), filename_end=str(end))
    model.load_state_dict(avg, strict=True)

    b0 = torch.load(start, map_location="cpu", weights_only=False)["batch_idx_train"]
    b1 = torch.load(end, map_location="cpu", weights_only=False)["batch_idx_train"]
    out = args.exp_dir / f"epoch-{args.epoch}-avg-{args.avg}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epoch,
            "avg": args.avg,
            "batch_idx_train_range": (b0, b1),
            "average_period": params.average_period,
        },
        out,
    )
    logging.info(
        f"batches ({b0}, {b1}] -> {(b1 - b0) // params.average_period} snapshots; saved {out}"
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    main()
