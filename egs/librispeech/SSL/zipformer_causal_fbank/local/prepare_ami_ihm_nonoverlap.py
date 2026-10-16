#!/usr/bin/env python3
"""
Cut single-speaker (non-overlapped) speech out of the AMI individual headset
(IHM) recordings, for self-supervised pre-training (train) and validation (dev).

Rules
  - Splits: AMI full-corpus-asr train (136 meetings) and dev (18 meetings).
    test and meetings outside the partition (IB4005) are NOT used.
  - Segments and word times: manual (NXT) annotations, parsed with
    load_segments() of icefall egs/ami/TSOT/local/prepare_utt_groups.py
    (segment extent = first word start .. last word end; non-word events
    ignored; a word with end < start is repaired).
  - Overlap removal at word boundaries: inside each manual segment, a word is
    dropped if it overlaps (strictly) any segment of another speaker; every
    maximal run of remaining consecutive words becomes a piece spanning
    first-word start .. last-word end.
  - Pieces longer than --max-dur (20 s) are split into ceil(dur / max_dur)
    parts at the word boundaries closest to the equal-split points (repeated
    on any part that is still too long).
  - Pieces shorter than --min-dur (1 s) are dropped.
  - Audio: the speaker's own headset <meeting>/audio/<meeting>.Headset-<ch>.wav,
    where <ch> is the `channel` of the speaker in corpusResources/meetings.xml
    (looked up by meeting + nxt_agent). Samples are copied unchanged
    (16 kHz, mono, PCM 16-bit) and verified by reading the output back.

Output: <out-dir>/<split>/wav/<meeting>/<id>.wav and <out-dir>/<split>/segments.tsv
  id = <meeting>-<speaker>-<start_cs:07d>-<end_cs:07d>  (cs = centiseconds)
"""
import argparse
import csv
import math
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from lhotse.recipes.ami import PARTITIONS

# egs/ami/TSOT/local/prepare_utt_groups.py
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "ami/TSOT/local"))
import prepare_utt_groups as PUG  # noqa: E402

SR = 16000


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ami-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev"])
    p.add_argument("--min-dur", type=float, default=1.0)
    p.add_argument("--max-dur", type=float, default=20.0)
    p.add_argument("--nj", type=int, default=32)
    return p.parse_args()


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def split_long(words, max_dur):
    """words: list of Word (consecutive). Returns list of word lists, each
    spanning <= max_dur where possible, cut at word boundaries."""
    dur = words[-1].end - words[0].start
    if dur <= max_dur or len(words) < 2:
        return [words]
    k = math.ceil(dur / max_dur)
    bounds = [w.end for w in words[:-1]]  # cut after word i
    cuts = []
    for j in range(1, k):
        target = words[0].start + dur * j / k
        i = int(np.argmin([abs(b - target) for b in bounds]))
        if not cuts or i > cuts[-1]:
            cuts.append(i)
    parts, prev = [], 0
    for i in cuts:
        parts.append(words[prev : i + 1])
        prev = i + 1
    parts.append(words[prev:])
    out = []
    for p in parts:
        out.extend(split_long(p, max_dur) if p[-1].end - p[0].start > max_dur else [p])
    return out


def pieces_of_meeting(segs, max_dur):
    """Yields (speaker, [words]) pieces of non-overlapped speech."""
    for s in segs:
        others = merge(
            [(o.start, o.end) for o in segs
             if o.speaker != s.speaker and o.start < s.end and s.start < o.end]
        )
        run = []
        for w in s.words + [None]:
            ok = w is not None and not any(a < w.end and w.start < b for a, b in others)
            if ok:
                run.append(w)
            elif run:
                for part in split_long(run, max_dur):
                    yield s.speaker, part
                run = []


def process_meeting(job):
    meeting, split, segs, spk2ch, args = job
    rows, stats = [], {"pieces": 0, "short": 0, "beyond_audio": 0}
    headsets = {}
    for speaker, words in pieces_of_meeting(segs, args.max_dur):
        start, end = words[0].start, words[-1].end
        if end - start < args.min_dur:
            stats["short"] += 1
            continue
        agent, ch = spk2ch[speaker]
        src = args.ami_dir / meeting / "audio" / f"{meeting}.Headset-{ch}.wav"
        if ch not in headsets:
            info = sf.info(str(src))
            assert info.samplerate == SR and info.channels == 1 and info.subtype == "PCM_16", src
            headsets[ch] = sf.read(str(src), dtype="int16")[0]
        audio = headsets[ch]
        a, b = int(round(start * SR)), int(round(end * SR))
        if b > len(audio):
            stats["beyond_audio"] += 1
            continue
        seg_id = f"{meeting}-{speaker}-{round(start * 100):07d}-{round(end * 100):07d}"
        out = args.out_dir / split / "wav" / meeting / f"{seg_id}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        x = audio[a:b]
        sf.write(str(out), x, SR, subtype="PCM_16", format="WAV")
        y, sr = sf.read(str(out), dtype="int16")
        assert sr == SR and y.ndim == 1 and np.array_equal(x, y), out
        text = " ".join(w.text for w in words)
        norm = " ".join(t for w in words for t in PUG.normalize_text_ami(w.text, normalize="kaldi").split())
        rows.append([seg_id, str(out.relative_to(args.out_dir / split)), meeting, speaker, agent, ch,
                     f"{start:.3f}", f"{end:.3f}", f"{end - start:.3f}", len(x), text, norm])
        stats["pieces"] += 1
    return meeting, rows, stats


def main():
    args = get_args()
    A = args.ami_dir / "ami_public_manual_1.6.2"
    meetings = PUG.load_segments(A, PUG.load_speaker_map(A))
    spk2ch = {}  # (meeting, global speaker) -> (agent, headset channel)
    for mt in ET.parse(A / "corpusResources" / "meetings.xml").getroot():
        for sp in mt:
            spk2ch[(mt.attrib["observation"], sp.attrib["global_name"])] = (
                sp.attrib["nxt_agent"], int(sp.attrib["channel"]))
    header = ["id", "wav", "meeting", "speaker", "agent", "headset", "start", "end",
              "duration", "num_samples", "text", "text_norm"]
    for split in args.splits:
        assert split in ("train", "dev"), "only train/dev may be used"
        jobs = [(m, split, meetings[m], {g: spk2ch[(m, g)] for g in {s.speaker for s in meetings[m]}}, args)
                for m in sorted(PARTITIONS["full-corpus-asr"][split])]
        all_rows, tot = [], {}
        with ProcessPoolExecutor(args.nj) as ex:
            for m, rows, st in ex.map(process_meeting, jobs):
                all_rows += rows
                for k, v in st.items():
                    tot[k] = tot.get(k, 0) + v
        all_rows.sort()
        with open(args.out_dir / split / "segments.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(header)
            w.writerows(all_rows)
        hours = sum(float(r[8]) for r in all_rows) / 3600
        print(f"{split}: {len(jobs)} meetings, {len(all_rows)} pieces, {hours:.2f} h, stats {tot}", flush=True)


if __name__ == "__main__":
    main()
