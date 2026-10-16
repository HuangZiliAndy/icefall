#!/usr/bin/env python3
"""
Convert the Kaldi-style utterance group directories created by
local/prepare_utt_groups.py into lhotse manifests with t-SOT transcripts.

For each utterance group one supervision is created that covers the whole
group (icefall's train.py uses the first supervision's text as the target):
  text      t-SOT transcript: all words of the group sorted by start time
            (ties: end time, then speaker), with "<cc>" inserted whenever the
            speaker changes. With <= 2 speakers per group, speaker change ==
            virtual channel change.
  speaker   "+"-joined sorted speaker ids
  alignment {"word": [...]} word times (same order as the t-SOT words)
  custom    word_speakers   speaker of each word in alignment["word"]
            text_by_speaker {speaker: words in time order} (for scoring)
            num_speakers, max_concurrent, left_cut, right_cut (from utt2info)

Groups shorter than --min-dur seconds are dropped.

Outputs (in --output-dir):
  ami_ihm_mix_utterance_group_recordings_<split>.jsonl.gz
  ami_ihm_mix_utterance_group_supervisions_<split>.jsonl.gz

Usage:
  python local/prepare_tsot_manifests.py \
      --kaldi-dir data/ihm_mix_utterance_group --output-dir data/manifests \
      --splits train dev --min-dur 0.5
  python local/prepare_tsot_manifests.py \
      --kaldi-dir data/ihm_mix_utterance_group --output-dir data/manifests \
      --splits test --min-dur 0
"""
import argparse
import logging
from collections import defaultdict
from pathlib import Path

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet
from lhotse.supervision import AlignmentItem

PREFIX = "ami_ihm_mix_utterance_group"
CC = "<cc>"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaldi-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", type=str, nargs="+", required=True)
    parser.add_argument("--min-dur", type=float, default=0.0)
    return parser.parse_args()


def read_kv(path: Path):
    with open(path) as f:
        return dict(line.strip().split(maxsplit=1) for line in f if line.strip())


def tsot_text(words):
    """words: list of (start, dur, speaker, token) sorted in t-SOT order."""
    out, prev = [], None
    for _, _, spk, tok in words:
        if prev is not None and spk != prev:
            out.append(CC)
        out.append(tok)
        prev = spk
    return " ".join(out)


def process_split(kaldi_dir: Path, output_dir: Path, split: str, min_dur: float):
    d = kaldi_dir / split
    wavs = read_kv(d / "wav.scp")
    reco2dur = {k: float(v) for k, v in read_kv(d / "reco2dur").items()}

    recordings = []
    for meeting, path in sorted(wavs.items()):
        rec = Recording.from_file(path, recording_id=meeting)
        assert rec.num_channels == 1, f"{path}: {rec.num_channels} channels"
        assert abs(rec.duration - reco2dur[meeting]) < 1e-2, (meeting, rec.duration)
        recordings.append(rec)

    ctm = defaultdict(list)
    with open(d / "words.ctm") as f:
        for line in f:
            gid, spk, st, dur, tok = line.split()
            ctm[gid].append((float(st), float(dur), spk, tok))

    info = {}
    with open(d / "utt2info") as f:
        for line in f:
            gid, dur, nspk, max_conc, left, right, nwords = line.split()
            info[gid] = dict(
                num_speakers=int(nspk), max_concurrent=int(max_conc),
                left_cut=left, right_cut=right, num_words=int(nwords),
            )

    sups, dropped = [], 0
    with open(d / "segments") as f:
        for line in f:
            gid, meeting, start, end = line.split()
            start, end = float(start), float(end)
            if end - start < min_dur:
                dropped += 1
                continue
            # t-SOT order: start time, then end time, then speaker
            words = sorted(ctm[gid], key=lambda w: (w[0], w[0] + w[1], w[2]))
            assert len(words) == info[gid]["num_words"], gid
            speakers = sorted({w[2] for w in words})
            assert len(speakers) == info[gid]["num_speakers"], gid
            by_spk = defaultdict(list)
            for _, _, spk, tok in words:
                by_spk[spk].append(tok)
            sups.append(
                SupervisionSegment(
                    id=gid,
                    recording_id=meeting,
                    start=start,
                    duration=round(end - start, 4),
                    channel=0,
                    text=tsot_text(words),
                    language="English",
                    speaker="+".join(speakers),
                    alignment={
                        "word": [
                            AlignmentItem(symbol=tok, start=round(start + st, 4), duration=dur)
                            for st, dur, _, tok in words
                        ]
                    },
                    custom={
                        "word_speakers": [w[2] for w in words],
                        "text_by_speaker": {s: " ".join(t) for s, t in sorted(by_spk.items())},
                        **{k: v for k, v in info[gid].items() if k != "num_words"},
                    },
                )
            )

    recordings = RecordingSet.from_recordings(recordings)
    supervisions = SupervisionSet.from_segments(sups)
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings.to_file(output_dir / f"{PREFIX}_recordings_{split}.jsonl.gz")
    supervisions.to_file(output_dir / f"{PREFIX}_supervisions_{split}.jsonl.gz")
    tot = sum(s.duration for s in sups)
    logging.info(
        f"{split}: {len(recordings)} recordings, {len(sups)} supervisions "
        f"({tot / 3600:.2f} h), dropped {dropped} groups shorter than {min_dur}s"
    )


def main():
    args = get_args()
    for split in args.splits:
        process_split(args.kaldi_dir, args.output_dir, split, args.min_dur)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
