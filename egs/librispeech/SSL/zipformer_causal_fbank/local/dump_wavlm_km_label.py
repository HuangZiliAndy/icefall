#!/usr/bin/env python3
"""
Assign k-means labels to every frame of WavLM-layer features without storing
the features: features are computed on the GPU exactly as in
local/dump_wavlm_feature.py (one utterance at a time, raw waveform, no
normalisation, output of the `layer`-th transformer layer, 50 Hz) and labelled
with the same nearest-centroid rule as fairseq simple_kmeans/dump_km_label.py
(ApplyKmeans): argmin_k ||x||^2 - 2 x.C_k + ||C_k||^2.

Utterances are processed one at a time on purpose: WavLM Base+'s first conv
layer uses GroupNorm over time, so padding inside a batch would change the
features w.r.t. those the k-means model was fitted on.

For each <name> in --sets, shard <rank> of <nshard> of <tsv_dir>/<name>.tsv is
written to <out_dir>/<name>_<rank>_<nshard>.km (one line of space-separated
labels per utterance, in tsv order), as fairseq does.

Usage:
  python local/dump_wavlm_km_label.py --tsv-dir D --out-dir D --sets a b \
      --ckpt WavLM-Base+.pt --layer 9 --centers km500_centers.npy --nshard 8 --rank 0
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# WavLM.py + modules.py from https://github.com/microsoft/unilm/tree/master/wavlm
# (commit 1ad6ea07df17b3e5c4c5d851f98721ecc43cf593); WAVLM_DIR = that directory
sys.path.insert(0, os.environ.get("WAVLM_DIR", "download/unilm/wavlm"))
from WavLM import WavLM, WavLMConfig  # noqa: E402

torch.set_num_threads(2)
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--centers", type=Path, required=True)
    p.add_argument("--nshard", type=int, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--max-chunk", type=int, default=1600000)
    return p.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = WavLMConfig(ck["cfg"])
    model = WavLM(cfg)
    model.load_state_dict(ck["model"])
    model.eval().to(dev)
    C = torch.from_numpy(np.load(args.centers)).to(dev)  # (K, D)
    Cnorm = (C ** 2).sum(1)

    for name in args.sets:
        with open(args.tsv_dir / f"{name}.tsv") as f:
            root = Path(f.readline().rstrip())
            lines = [line.rstrip() for line in f]
        tot = len(lines)
        start, end = round(tot * args.rank / args.nshard), round(tot * (args.rank + 1) / args.nshard)
        out = args.out_dir / f"{name}_{args.rank}_{args.nshard}.km"
        tmp = out.with_suffix(".km.tmp")
        t0 = time.time()
        with open(tmp, "w") as fout:
            for k, line in enumerate(lines[start:end]):
                path, nsample = line.split("\t")
                wav, sr = sf.read(str(root / path), dtype="float32")
                assert sr == 16000 and wav.ndim == 1 and len(wav) == int(nsample), path
                x = torch.from_numpy(wav).to(dev)[None]
                if cfg.normalize:
                    x = torch.nn.functional.layer_norm(x, x.shape)
                feats = [
                    model.extract_features(x[:, s:s + args.max_chunk], output_layer=args.layer)[0].squeeze(0)
                    for s in range(0, x.size(1), args.max_chunk)
                ]
                feat = torch.cat(feats, 0).float()
                dist = (feat ** 2).sum(1, keepdim=True) - 2 * feat @ C.T + Cnorm
                lab = dist.argmin(1).cpu().numpy()
                fout.write(" ".join(map(str, lab)) + "\n")
                if (k + 1) % 5000 == 0:
                    logging.info(f"{name}: {k + 1}/{end - start} ({time.time() - t0:.0f}s)")
        tmp.rename(out)
        logging.info(f"{name}: done {end - start} utterances ({start}-{end}) in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
