#!/usr/bin/env python3
"""
Split an utterance-group set (e.g. test) into subsets by number of speakers,
both as Kaldi-style directories and as lhotse cut manifests.

The number of speakers of a group is column 3 of utt2info (same as
supervision.custom["num_speakers"] in the cuts). Features are not recomputed:
the subset cuts point to the same feature storage.

Outputs, for each N in --num-speakers:
  <kaldi-dir>/<split>_<N>spk/   segments text utt2spk spk2utt wav.scp reco2dur
                                words.ctm utt2info (filtered)
  <fbank-dir>/ami_ihm_mix_utterance_group_cuts_<split>_<N>spk.jsonl.gz

Usage:
  python local/subset_by_num_speakers.py --split test --num-speakers 1 2
"""
import argparse
import logging
from collections import defaultdict
from pathlib import Path

from lhotse import load_manifest

PREFIX = "ami_ihm_mix_utterance_group"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-speakers", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--kaldi-dir", type=Path, default=Path("data/ihm_mix_utterance_group")
    )
    parser.add_argument("--fbank-dir", type=Path, default=Path("data/fbank"))
    return parser.parse_args()


def filter_lines(src: Path, dst: Path, keep) -> int:
    n = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if keep(line.split(maxsplit=1)[0]):
                fout.write(line)
                n += 1
    return n


def main():
    args = get_args()
    src = args.kaldi_dir / args.split

    nspk = {}
    with open(src / "utt2info") as f:
        for line in f:
            fields = line.split()
            nspk[fields[0]] = int(fields[2])
    utt2reco = {}
    with open(src / "segments") as f:
        for line in f:
            gid, reco = line.split()[:2]
            utt2reco[gid] = reco

    cuts = load_manifest(args.fbank_dir / f"{PREFIX}_cuts_{args.split}.jsonl.gz")

    for n in args.num_speakers:
        name = f"{args.split}_{n}spk"
        utts = {u for u, k in nspk.items() if k == n}
        recos = {utt2reco[u] for u in utts}

        # Kaldi dir
        dst = args.kaldi_dir / name
        dst.mkdir(parents=True, exist_ok=True)
        for fname in ["segments", "text", "utt2spk", "words.ctm", "utt2info"]:
            filter_lines(src / fname, dst / fname, lambda k: k in utts)
        for fname in ["wav.scp", "reco2dur"]:
            filter_lines(src / fname, dst / fname, lambda k: k in recos)
        spk2utt = defaultdict(list)
        with open(dst / "utt2spk") as f:
            for line in f:
                u, s = line.split()
                spk2utt[s].append(u)
        with open(dst / "spk2utt", "w") as f:
            for s in sorted(spk2utt):
                f.write(f"{s} {' '.join(spk2utt[s])}\n")

        # lhotse cuts
        sub = cuts.filter(lambda c: c.supervisions[0].custom["num_speakers"] == n)
        sub = sub.to_eager()
        assert {c.id for c in sub} == utts, name
        out = args.fbank_dir / f"{PREFIX}_cuts_{name}.jsonl.gz"
        sub.to_file(out)

        hours = sum(c.duration for c in sub) / 3600
        logging.info(
            f"{name}: {len(utts)} groups, {len(recos)} meetings, {hours:.2f} h "
            f"-> {dst}, {out}"
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
