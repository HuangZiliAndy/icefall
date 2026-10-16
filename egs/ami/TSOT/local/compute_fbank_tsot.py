#!/usr/bin/env python3
"""
Compute 80-dim fbank features for the AMI IHM-Mix utterance group manifests
created by local/prepare_tsot_manifests.py.

One cut per utterance group (cut id = group id) is cut out of the meeting
recordings. With --perturb-speed, 0.9x and 1.1x speed-perturbed copies are
added (use for train only).

Inputs:  data/manifests/ami_ihm_mix_utterance_group_{recordings,supervisions}_<split>.jsonl.gz
Outputs: data/fbank/ami_ihm_mix_utterance_group_cuts_<split>.jsonl.gz
         data/fbank/ami_ihm_mix_utterance_group_feats_<split>/

Usage:
  python local/compute_fbank_tsot.py --split train --perturb-speed 1
  python local/compute_fbank_tsot.py --split dev
  python local/compute_fbank_tsot.py --split test
"""
import argparse
import logging
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, load_manifest

from icefall.utils import str2bool

PREFIX = "ami_ihm_mix_utterance_group"

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--perturb-speed", type=str2bool, default=False)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fbank"))
    parser.add_argument("--num-jobs", type=int, default=32)
    return parser.parse_args()


def main():
    args = get_args()
    cuts_file = args.output_dir / f"{PREFIX}_cuts_{args.split}.jsonl.gz"
    if cuts_file.is_file():
        logging.info(f"{cuts_file} exists - skipping")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recordings = load_manifest(
        args.manifest_dir / f"{PREFIX}_recordings_{args.split}.jsonl.gz"
    )
    supervisions = load_manifest(
        args.manifest_dir / f"{PREFIX}_supervisions_{args.split}.jsonl.gz"
    )
    cuts = CutSet.from_manifests(
        recordings=recordings, supervisions=supervisions
    ).trim_to_supervisions(keep_overlapping=False)
    cuts = cuts.to_eager()
    assert len(cuts) == len(supervisions), (len(cuts), len(supervisions))
    for c in cuts:
        assert len(c.supervisions) == 1, c.id
    logging.info(f"{args.split}: {len(cuts)} cuts")

    if args.perturb_speed:
        logging.info("Doing speed perturb (0.9, 1.1)")
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)

    extractor = Fbank(FbankConfig(num_mel_bins=80))
    cuts = cuts.compute_and_store_features(
        extractor=extractor,
        storage_path=f"{args.output_dir}/{PREFIX}_feats_{args.split}",
        num_jobs=args.num_jobs,
        storage_type=LilcomChunkyWriter,
    )
    cuts.to_file(cuts_file)
    logging.info(f"Saved {len(cuts)} cuts to {cuts_file}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
