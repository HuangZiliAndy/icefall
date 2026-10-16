#!/usr/bin/env python3
"""
Dump features of a WavLM transformer layer in the fairseq simple_kmeans format,
so that fairseq's examples/hubert/simple_kmeans/learn_kmeans.py and
dump_km_label.py can be used unchanged.

Mirrors fairseq examples/hubert/simple_kmeans/dump_hubert_feature.py, with the
model loaded by the official WavLM code (microsoft/unilm wavlm/WavLM.py):
  - input: raw 16 kHz waveform, no normalisation (WavLM Base+ cfg.normalize=False)
  - feature: output of the `layer`-th transformer layer (1-based; same as
    fairseq's output_layer), 50 Hz
  - shard i of n covers utterances [round(N*i/n), round(N*(i+1)/n)) of the tsv
  - output: <feat_dir>/<split>_<rank>_<nshard>.npy (float32, frames x dim,
    utterances concatenated) and .len (frames per utterance, one per line)

Usage:
  python local/dump_wavlm_feature.py <tsv_dir> <split> <ckpt> <layer> <nshard> <rank> <feat_dir>
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# WavLM.py + modules.py from https://github.com/microsoft/unilm/tree/master/wavlm
# (commit 1ad6ea07df17b3e5c4c5d851f98721ecc43cf593); WAVLM_DIR = that directory
sys.path.insert(0, os.environ.get("WAVLM_DIR", "download/unilm/wavlm"))
from WavLM import WavLM, WavLMConfig  # noqa: E402

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("tsv_dir")
    p.add_argument("split")
    p.add_argument("ckpt_path")
    p.add_argument("layer", type=int)
    p.add_argument("nshard", type=int)
    p.add_argument("rank", type=int)
    p.add_argument("feat_dir")
    p.add_argument("--max_chunk", type=int, default=1600000)  # as in fairseq
    return p.parse_args()


def get_shard(tsv, nshard, rank):
    with open(tsv) as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
    tot = len(lines)
    start, end = round(tot * rank / nshard), round(tot * (rank + 1) / nshard)
    logging.info(f"rank {rank} of {nshard}, process {end - start} ({start}-{end}) out of {tot}")
    return root, lines[start:end]


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda")
    ck = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    cfg = WavLMConfig(ck["cfg"])
    model = WavLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    assert 1 <= args.layer <= cfg.encoder_layers

    root, lines = get_shard(Path(args.tsv_dir) / f"{args.split}.tsv", args.nshard, args.rank)
    feats, lens = [], []
    for k, line in enumerate(lines):
        path, nsample = line.split("\t")
        wav, sr = sf.read(str(Path(root) / path), dtype="float32")
        assert sr == 16000 and wav.ndim == 1 and len(wav) == int(nsample), path
        x = torch.from_numpy(wav).to(device)[None]
        if cfg.normalize:
            x = torch.nn.functional.layer_norm(x, x.shape)
        chunks = []
        for start in range(0, x.size(1), args.max_chunk):
            rep = model.extract_features(x[:, start:start + args.max_chunk], output_layer=args.layer)[0]
            chunks.append(rep.squeeze(0))
        feat = torch.cat(chunks, 0).float().cpu().numpy()
        assert np.isfinite(feat).all(), path
        feats.append(feat)
        lens.append(len(feat))
        if (k + 1) % 1000 == 0:
            logging.info(f"{k + 1}/{len(lines)} utterances")

    out = Path(args.feat_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / f"{args.split}_{args.rank}_{args.nshard}"
    np.save(f"{prefix}.npy", np.concatenate(feats, 0))
    with open(f"{prefix}.len", "w") as f:
        f.write("\n".join(map(str, lens)) + "\n")
    logging.info(f"finished: {len(lens)} utterances, {sum(lens)} frames -> {prefix}.npy")


if __name__ == "__main__":
    main()
