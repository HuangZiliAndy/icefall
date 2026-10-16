#!/usr/bin/env python3
"""
Build lhotse CutSets for SSL pre-training from the k-means label directory
(<set>.tsv + <set>.km, see local/dump_wavlm_km_label.py).

Each cut covers one whole 16 kHz mono WAV file and carries its frame labels
(50 Hz, WavLM frames: floor((num_samples - 400) / 320) + 1 labels) as a string
in cut.custom["kmeans"], the format read by the k2SSL dataset. No features
are stored: fbank is computed on the fly (per cut) at training time.

Outputs (in --out-dir):
  ssl_cuts_<set>.jsonl.gz       for every set
  ssl_cuts_train-shuf.jsonl.gz  LibriSpeech 960 h + AMI IHM train, shuffled
                                (natural proportions, no up-sampling)

Usage:
  python local/prepare_ssl_manifests.py \
      --label-dir data/kmeans/wavlm_base_plus_L9_km500/labels --out-dir data/manifests
"""
import argparse
import logging
import random
from pathlib import Path

from lhotse import CutSet, MonoCut, Recording
from lhotse.audio import AudioSource

TRAIN_SETS = ["ls_train-clean-100", "ls_train-clean-360", "ls_train-other-500", "ami_ihm_train"]
DEV_SETS = ["ls_dev-clean", "ls_dev-other", "ami_ihm_dev"]
SR = 16000


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_set(label_dir: Path, name: str) -> CutSet:
    with open(label_dir / f"{name}.tsv") as f:
        root = Path(f.readline().rstrip())
        entries = [line.rstrip().split("\t") for line in f]
    with open(label_dir / f"{name}.km") as f:
        labels = [line.rstrip() for line in f]
    assert len(entries) == len(labels), name
    cuts = []
    for (rel, nsample), lab in zip(entries, labels):
        n = int(nsample)
        assert len(lab.split()) == (n - 400) // 320 + 1, (name, rel)
        path = root / rel
        rid = f"{name}/{Path(rel).with_suffix('').as_posix()}"
        rec = Recording(
            id=rid,
            sources=[AudioSource(type="file", channels=[0], source=str(path))],
            sampling_rate=SR,
            num_samples=n,
            duration=n / SR,
        )
        cuts.append(MonoCut(id=rid, start=0.0, duration=n / SR, channel=0,
                            recording=rec, custom={"kmeans": lab}))
    return CutSet.from_cuts(cuts)


def main():
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = []
    for name in TRAIN_SETS + DEV_SETS:
        cs = load_set(args.label_dir, name)
        cs.to_file(args.out_dir / f"ssl_cuts_{name}.jsonl.gz")
        hours = sum(c.duration for c in cs) / 3600
        logging.info(f"{name}: {len(cs)} cuts, {hours:.2f} h")
        if name in TRAIN_SETS:
            train.extend(cs)
    random.Random(args.seed).shuffle(train)
    CutSet.from_cuts(train).to_file(args.out_dir / "ssl_cuts_train-shuf.jsonl.gz")
    logging.info(f"train-shuf: {len(train)} cuts, {sum(c.duration for c in train) / 3600:.2f} h")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    main()
