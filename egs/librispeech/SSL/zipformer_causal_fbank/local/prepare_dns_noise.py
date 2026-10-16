#!/usr/bin/env python3
"""
Prepare the DNS Challenge 3 (wideband) noise set for WavLM-style noisy speech
simulation (zipformer/wavlm_augment.py).

All output files are 16 kHz, mono, PCM-16 WAV. Files that already have this
format are hard-linked from --in-dir (no extra space); the others (188 files
at 44.1 / 48 / 96 kHz, all mono PCM-16) are resampled with
torchaudio.functional.resample and written as new files. --in-dir is not
modified.

Also writes a fairseq-style list (first line: root, then
"<file name>\\t<num samples>") used by the data loader.

Usage:
  python local/prepare_dns_noise.py \
      --in-dir /workspace/dataset/DNS_Challenge_3/datasets/noise \
      --out-dir /workspace/dataset/DNS_Challenge_3/noise_16k \
      --list data/noise/dns_noise.tsv
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR = 16000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--list", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.list.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in os.listdir(args.in_dir) if f.endswith(".wav"))
    n_link = n_conv = 0
    rows = []
    for f in files:
        src, dst = args.in_dir / f, args.out_dir / f
        info = sf.info(str(src))
        if (info.samplerate, info.channels, info.subtype) == (SR, 1, "PCM_16"):
            if not dst.exists():
                os.link(src, dst)
            n_link += 1
            rows.append((f, info.frames))
            continue
        wav, sr = sf.read(str(src), dtype="float32", always_2d=True)  # (T, C)
        wav = wav.mean(axis=1)
        if sr != SR:
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
        wav = np.clip(wav, -1.0, 32767 / 32768)
        sf.write(str(dst), wav, SR, subtype="PCM_16")
        n_conv += 1
        rows.append((f, len(wav)))
        logging.info(f"converted {f}: {sr} Hz, {info.channels} ch -> 16 kHz mono")

    with open(args.list, "w") as fout:
        fout.write(f"{args.out_dir}\n")
        for f, n in rows:
            fout.write(f"{f}\t{n}\n")
    hours = sum(n for _, n in rows) / SR / 3600
    logging.info(f"{len(rows)} files ({n_link} linked, {n_conv} converted), {hours:.1f} h -> {args.list}")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    main()
