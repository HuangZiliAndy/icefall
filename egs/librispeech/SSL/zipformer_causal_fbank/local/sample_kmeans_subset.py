#!/usr/bin/env python3
"""
Sample the utterances used to fit the k-means model (fairseq simple_kmeans
style: random sampling by number of utterances, not duration).

  LibriSpeech train (960 h, WAV)        : 90/960 of the utterances  (~90 h)
  AMI IHM non-overlap train (~49 h)     : 1/5 of the pieces         (~10 h)

dev sets are not used. Writes a fairseq-style tsv (first line: root dir,
then "<path relative to root>\\t<num samples>") and a summary.

Usage:
  python local/sample_kmeans_subset.py \
      --out-dir data/kmeans/wavlm_base_plus_L9_km500 --seed 0
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-dir", type=Path, default=Path("/workspace/dataset/LibriSpeech_wav"))
    p.add_argument("--ami-dir", type=Path, default=Path("/workspace/dataset/AMI_IHM_nonoverlap"))
    p.add_argument("--ls-fraction", type=float, default=90 / 960)
    p.add_argument("--ami-fraction", type=float, default=1 / 5)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)

    ls = sorted(
        p for s in ["train-clean-100", "train-clean-360", "train-other-500"]
        for p in (args.librispeech_dir / s).rglob("*.wav")
    )
    with open(args.ami_dir / "train" / "segments.tsv") as f:
        ami_rows = list(csv.DictReader(f, delimiter="\t"))
    ami = [(args.ami_dir / "train" / r["wav"], int(r["num_samples"])) for r in ami_rows]

    n_ls = math.ceil(len(ls) * args.ls_fraction)
    n_ami = math.ceil(len(ami) * args.ami_fraction)
    ls_pick = [ls[i] for i in sorted(rng.choice(len(ls), n_ls, replace=False))]
    ami_pick = [ami[i] for i in sorted(rng.choice(len(ami), n_ami, replace=False))]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tot = {"librispeech": [0, 0], "ami": [0, 0]}
    with open(args.out_dir / "kmeans_subset.tsv", "w") as f:
        f.write("/\n")
        for p in ls_pick:
            n = sf.info(str(p)).frames
            f.write(f"{str(p).lstrip('/')}\t{n}\n")
            tot["librispeech"][0] += 1
            tot["librispeech"][1] += n
        for p, n in ami_pick:
            f.write(f"{str(p).lstrip('/')}\t{n}\n")
            tot["ami"][0] += 1
            tot["ami"][1] += n

    lines = [f"seed={args.seed}",
             f"librispeech: {tot['librispeech'][0]} / {len(ls)} utterances "
             f"(fraction {args.ls_fraction:.4f}), {tot['librispeech'][1] / 16000 / 3600:.2f} h",
             f"ami: {tot['ami'][0]} / {len(ami)} pieces "
             f"(fraction {args.ami_fraction:.4f}), {tot['ami'][1] / 16000 / 3600:.2f} h",
             f"total: {(tot['librispeech'][1] + tot['ami'][1]) / 16000 / 3600:.2f} h"]
    (args.out_dir / "kmeans_subset_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
