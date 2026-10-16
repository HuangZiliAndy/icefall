#!/usr/bin/env python3
"""
Convert LibriSpeech FLAC (<src>) to 16 kHz mono PCM-16 WAV in <dst>, keeping
the directory layout. The source is already 16 kHz / mono / 16-bit, so this only changes the
container (lossless). Every written file is read back and compared sample by
sample with the FLAC. *.trans.txt and the top-level *.TXT files are copied.
Resumable: existing, verified outputs are skipped.

Usage: python local/convert_librispeech_to_wav.py <src> <dst> [nj]
"""
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])
NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 32
SUBSETS = ["dev-clean", "dev-other", "test-clean", "test-other",
           "train-clean-100", "train-clean-360", "train-other-500"]


def convert(src: Path) -> str:
    dst = DST / src.relative_to(SRC).with_suffix(".wav")
    x, sr = sf.read(src, dtype="int16")
    if sr != 16000 or x.ndim != 1:
        return f"BAD_INPUT {src} sr={sr} shape={x.shape}"
    if dst.is_file():
        try:
            y, sr2 = sf.read(dst, dtype="int16")
            if sr2 == 16000 and np.array_equal(x, y):
                return "SKIP"
        except Exception:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".wav.tmp")
    sf.write(tmp, x, 16000, subtype="PCM_16", format="WAV")
    y, sr2 = sf.read(tmp, dtype="int16")
    info = sf.info(tmp)
    if not (sr2 == 16000 and info.channels == 1 and info.subtype == "PCM_16"
            and np.array_equal(x, y)):
        return f"MISMATCH {src}"
    tmp.rename(dst)
    return "OK"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob("*.TXT"):
        shutil.copy2(f, DST / f.name)
    files = []
    for s in SUBSETS:
        for t in (SRC / s).rglob("*.trans.txt"):
            out = DST / t.relative_to(SRC)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(t, out)
        files += sorted((SRC / s).rglob("*.flac"))
    print(f"{len(files)} flac files", flush=True)
    counts = {}
    with ProcessPoolExecutor(NJ) as ex:
        for k, r in enumerate(ex.map(convert, files, chunksize=64), 1):
            key = r.split()[0]
            counts[key] = counts.get(key, 0) + 1
            if key not in ("OK", "SKIP"):
                print(r, flush=True)
            if k % 20000 == 0:
                print(f"{k}/{len(files)} {counts}", flush=True)
    print(f"DONE {counts}", flush=True)
    sys.exit(0 if set(counts) <= {"OK", "SKIP"} else 1)


if __name__ == "__main__":
    main()
