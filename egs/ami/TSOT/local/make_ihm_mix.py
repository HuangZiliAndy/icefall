#!/usr/bin/env python3
"""
Create IHM-Mix audio for AMI by summing all headset (IHM) channels of a meeting,
reproducing the official AMI Mix-Headset files.

The official <M>.Mix-Headset.wav (groups.inf.ed.ac.uk mirror, not on openslr)
was found to be 0.8 x the sum of the headset channels, with no delay and no
time-varying gain (fit on 6 meetings); the remaining +-2 LSB difference looks
like dither and cannot be reproduced exactly. So for each meeting <M> this
script reads <ami-dir>/<M>/audio/<M>.Headset-*.wav, pads them to the same
length, and writes <out-dir>/<M>.wav = clip(gain * sum) as 16 kHz mono PCM_16.
The number of clipped samples is logged.

Usage:
  python local/make_ihm_mix.py --ami-dir /workspace/dataset/AMI \
      --out-dir /workspace/dataset/AMI_IHM_MIX --nj 32
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ami-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gain", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nj", type=int, default=16)
    return parser.parse_args()


def mix_meeting(meeting_dir: Path, out_dir: Path, gain: float, overwrite: bool) -> str:
    meeting = meeting_dir.name
    out_path = out_dir / f"{meeting}.wav"
    headsets = sorted((meeting_dir / "audio").glob(f"{meeting}.Headset-*.wav"))
    if not headsets:
        return f"{meeting}: no headset files, skipped"
    if out_path.exists() and not overwrite:
        return f"{meeting}: exists, skipped"

    signals = []
    for h in headsets:
        x, sr = sf.read(h, dtype="int16", always_2d=False)
        assert sr == 16000, f"{h}: sampling rate {sr}"
        assert x.ndim == 1, f"{h}: expected mono, got shape {x.shape}"
        signals.append(x)

    lengths = [len(x) for x in signals]
    mix = np.zeros(max(lengths), dtype=np.int64)
    for x in signals:
        mix[: len(x)] += x

    mix = np.round(gain * mix)
    nclip = int(np.sum((mix > 32767) | (mix < -32768)))
    mix = np.clip(mix, -32768, 32767).astype(np.int16)

    tmp_path = out_path.with_suffix(".wav.tmp")
    sf.write(tmp_path, mix, 16000, subtype="PCM_16", format="WAV")
    tmp_path.rename(out_path)
    return (
        f"{meeting}: {len(headsets)} headsets, {max(lengths) / 16000:.2f}s, "
        f"length diff {(max(lengths) - min(lengths)) / 16000:.3f}s, "
        f"clipped {nclip} samples ({nclip / len(mix) * 100:.5f}%)"
    )


def main():
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meeting_dirs = sorted(
        d for d in args.ami_dir.iterdir() if d.is_dir() and (d / "audio").is_dir()
    )
    logging.info(f"Found {len(meeting_dirs)} meeting directories, gain {args.gain}")
    with ProcessPoolExecutor(args.nj) as ex:
        futures = [
            ex.submit(mix_meeting, d, args.out_dir, args.gain, args.overwrite)
            for d in meeting_dirs
        ]
        for f in futures:
            logging.info(f.result())
    logging.info("Done")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
