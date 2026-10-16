#!/usr/bin/env python3
"""
Write fairseq-style tsv files (first line: root, then "<rel path>\\t<num samples>")
for every set that gets k-means labels:

  ls_train-clean-100, ls_train-clean-360, ls_train-other-500, ls_dev-clean,
  ls_dev-other   (LibriSpeech, WAV)
  ami_ihm_train, ami_ihm_dev   (AMI IHM non-overlap pieces)

Usage: python local/make_label_tsvs.py --out-dir data/kmeans/wavlm_base_plus_L9_km500/labels
"""
import argparse
import csv
from pathlib import Path

import soundfile as sf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-dir", type=Path, required=True)
    p.add_argument("--ami-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for s in ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other"]:
        root = args.librispeech_dir / s
        files = sorted(root.rglob("*.wav"))
        with open(args.out_dir / f"ls_{s}.tsv", "w") as f:
            f.write(f"{root}\n")
            for w in files:
                f.write(f"{w.relative_to(root)}\t{sf.info(str(w)).frames}\n")
        print(f"ls_{s}: {len(files)}")

    for s in ["train", "dev"]:
        root = args.ami_dir / s
        with open(root / "segments.tsv") as fin:
            rows = sorted(csv.DictReader(fin, delimiter="\t"), key=lambda r: r["id"])
        with open(args.out_dir / f"ami_ihm_{s}.tsv", "w") as f:
            f.write(f"{root}\n")
            for r in rows:
                f.write(f"{r['wav']}\t{r['num_samples']}\n")
        print(f"ami_ihm_{s}: {len(rows)}")


if __name__ == "__main__":
    main()
