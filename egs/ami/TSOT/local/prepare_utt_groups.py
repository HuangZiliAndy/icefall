#!/usr/bin/env python3
"""
Prepare Kaldi-style "utterance group" data directories for AMI from the
manual (NXT) annotations.

An utterance group is a maximal stretch of speech with no silence in it: all
speaker segments of a meeting (across speakers) are put on one time line and
segments that overlap or touch (gap <= 0) are merged into one group. A group
may therefore contain several sentences from several speakers.

Manual segments are atomic, so a group is never cut inside a sentence unless it
is longer than --max-dur. Such groups are split by dynamic programming at
"safe" cut points, i.e. points where nobody else is speaking:
  level 1: turn change with no overlap, or a sentence end (. ? !) of the only
           active speaker
  level 2: a comma (or other punctuation) of the only active speaker
  level 3: any word boundary of the only active speaker
Level k is only used if no split with levels < k keeps every piece within
--max-dur. Among valid splits, the one with the fewest cuts is chosen, ties
broken by maximising the shortest piece. If even level 3 fails, cuts are made
greedily and the piece is allowed to exceed --max-dur (marked "forced").
With --max-dur 0 groups are never split (used for the test set, which keeps
all natural groups so that they can later be selected by number of speakers).

With --max-speakers N (used for train/dev, N=2), the split is instead done
jointly by speaker count and length: a piece is usable if it has <= N speakers
and lasts <= --max-dur. Using safe cut points of level <= --max-cut-level, a DP
maximises the words kept in usable pieces, then minimises #cuts, then
#word-level cuts, then #comma-level cuts. Unusable pieces are dropped.

Segment/word times come from the NXT words/*.words.xml files. Silence-bounded
group edges are padded by --pad seconds (at most half the gap to the
neighbouring group); cut edges are not padded.

Output per split (<out-dir>/<split>/):
  wav.scp     <meeting> <audio path>
  segments    <group_id> <meeting> <start> <end>
  text        <group_id> <words>   (segments ordered by start time, words kept
                                    contiguous within each segment)
  utt2spk     <group_id> <meeting>
  spk2utt     <meeting> <group_id> ...
  reco2dur    <meeting> <duration>
  words.ctm   <group_id> <speaker> <start> <dur> <word>  (times relative to
                                    the group start in `segments`)
  utt2info    <group_id> <dur> <nspk> <max_concurrent> <left_cut> <right_cut> <nwords>
              cut types: silence, turn, sentence, comma, word, forced

Usage:
  python local/prepare_utt_groups.py \
      --annotations-dir download/AMI/ami_public_manual_1.6.2 \
      --audio-dir download/AMI_IHM_MIX_OFFICIAL \
      --wav-template "{meeting}.Mix-Headset.wav" \
      --out-dir data/ihm_mix_utterance_group --splits train dev \
      --max-dur 25 --max-speakers 2 --max-cut-level 3
  python local/prepare_utt_groups.py \
      --annotations-dir download/AMI/ami_public_manual_1.6.2 \
      --audio-dir download/AMI_IHM_MIX_OFFICIAL \
      --wav-template "{meeting}.Mix-Headset.wav" \
      --out-dir data/ihm_mix_utterance_group --splits test --max-dur 0
"""
import argparse
import bisect
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
from lhotse.recipes.ami import PARTITIONS
from lhotse.recipes.utils import normalize_text_ami

NS = "{http://nite.sourceforge.net/}"
SENTENCE_END = {".", "?", "!"}
# Lower is better. "turn" and "sentence" share level 1.
CUT_LEVEL = {"turn": 1, "sentence": 1, "comma": 2, "word": 3}

stats = Counter()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument(
        "--wav-template", type=str, default="{meeting}.wav",
        help="Audio file name inside --audio-dir, e.g. {meeting}.Mix-Headset.wav",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--max-dur", type=float, default=25.0,
        help="Split groups longer than this (seconds); <= 0 disables splitting.",
    )
    parser.add_argument(
        "--max-speakers", type=int, default=0,
        help="If > 0, split jointly by speaker count and --max-dur, and drop "
        "pieces with more speakers (or longer than --max-dur) than allowed.",
    )
    parser.add_argument(
        "--max-cut-level", type=int, default=3, choices=[1, 2, 3],
        help="Highest cut level used by the joint split (--max-speakers > 0).",
    )
    parser.add_argument("--pad", type=float, default=0.1)
    parser.add_argument("--partition", type=str, default="full-corpus-asr")
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["train", "dev", "test"],
        help="Which splits of --partition to prepare.",
    )
    return parser.parse_args()


@dataclass
class Word:
    start: float
    end: float
    text: str  # raw NXT text


@dataclass
class Segment:
    speaker: str  # global speaker id, e.g. MEE006
    start: float
    end: float
    words: List[Word]
    # word-end time -> cut type ("sentence", "comma" or "word")
    boundaries: Dict[float, str] = field(default_factory=dict)


def load_speaker_map(annotations_dir: Path) -> Dict[Tuple[str, str], str]:
    """(meeting, nxt_agent) -> global speaker name."""
    tree = ET.parse(annotations_dir / "corpusResources" / "meetings.xml")
    spk_map = {}
    for meeting in tree.getroot():
        m = meeting.attrib["observation"]
        for spk in meeting:
            spk_map[(m, spk.attrib["nxt_agent"])] = spk.attrib["global_name"]
    return spk_map


def load_segments(
    annotations_dir: Path, spk_map: Dict[Tuple[str, str], str]
) -> Dict[str, List[Segment]]:
    meetings = defaultdict(list)
    for words_file in sorted((annotations_dir / "words").glob("*.words.xml")):
        meeting, agent, _ = words_file.name.split(".", 2)
        if (meeting, agent) not in spk_map:
            # A few files (e.g. EN2002c.D) belong to no speaker in meetings.xml
            # and are empty.
            assert len(ET.parse(words_file).getroot()) == 0, words_file
            stats["empty_words_files_without_speaker"] += 1
            continue
        speaker = spk_map[(meeting, agent)]

        # Tokens in file order: (kind, start, end, text); kind in w/p/e
        tokens = []
        id2idx = {}
        for el in ET.parse(words_file).getroot():
            id2idx[el.get(NS + "id")] = len(tokens)
            st, et = el.get("starttime"), el.get("endtime")
            st = float(st) if st is not None else None
            et = float(et) if et is not None else None
            if el.tag == "w":
                kind = "p" if el.get("punc") == "true" else "w"
            else:
                kind = "e"  # vocalsound, disfmarker, gap, ...
            tokens.append((kind, st, et, el.text))
        stats["words_total"] += sum(1 for t in tokens if t[0] == "w")

        seg_file = annotations_dir / "segments" / f"{meeting}.{agent}.segments.xml"
        for seg in ET.parse(seg_file).getroot():
            child = seg.find(NS + "child")
            if child is None:
                continue
            ids = re.findall(r"id\(([^)]+)\)", child.attrib["href"])
            seg_tokens = tokens[id2idx[ids[0]] : id2idx[ids[-1]] + 1]

            # Only words define the extent of a segment. Non-word events
            # (vocalsound, gap, ...) are ignored: some have implausible
            # durations (e.g. a 217 s gap in EN2001d) that would chain whole
            # stretches of a meeting into one group.
            words, boundaries = [], {}
            for kind, st, et, text in seg_tokens:
                if kind == "w":
                    if st is None or et is None:
                        stats["words_no_time_in_segment"] += 1
                        continue
                    if not normalize_text_ami(text, normalize="kaldi"):
                        stats["words_empty_after_norm"] += 1
                        continue
                    words.append(Word(st, et, text))
                elif kind == "p" and words:
                    t = words[-1].end
                    cut = "sentence" if text in SENTENCE_END else "comma"
                    if CUT_LEVEL[cut] < CUT_LEVEL.get(boundaries.get(t), 9):
                        boundaries[t] = cut
            if not words:
                continue
            # Repair words with end < start (e.g. EN2006a "do" 2162.31-2160.39):
            # end at the next word's start.
            for k, w in enumerate(words):
                if w.end < w.start:
                    nxt = words[k + 1].start if k + 1 < len(words) else w.start
                    w.end = max(w.start, nxt)
                    stats["words_negative_dur_repaired"] += 1
            for w in words:
                boundaries.setdefault(w.end, "word")

            start = min(w.start for w in words)
            end = max(w.end for w in words)
            # Only boundaries strictly inside the segment can be cut points.
            boundaries = {t: c for t, c in boundaries.items() if start < t < end}
            stats["words_in_segments"] += len(words)
            meetings[meeting].append(Segment(speaker, start, end, words, boundaries))
    return meetings


def find_cuts(start: float, end: float, cands: List[float], max_dur: float):
    """Fewest cuts such that all pieces <= max_dur; ties: maximise the shortest
    piece. Returns None if infeasible."""
    pts = [start] + cands + [end]
    n = len(pts)
    best: List[Optional[Tuple[int, float, int]]] = [None] * n  # (ncuts, minlen, prev)
    best[0] = (0, float("inf"), -1)
    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            length = pts[j] - pts[i]
            if length > max_dur:
                break
            if best[i] is None:
                continue
            ncuts = best[i][0] + (1 if j < n - 1 else 0)
            minlen = min(best[i][1], length)
            if best[j] is None or (ncuts, -minlen) < (best[j][0], -best[j][1]):
                best[j] = (ncuts, minlen, i)
    if best[-1] is None:
        return None
    cuts, j = [], n - 1
    while best[j][2] > 0:
        j = best[j][2]
        cuts.append(pts[j])
    return sorted(cuts)


def greedy_cuts(start: float, end: float, cands: List[float], max_dur: float):
    """Fallback: take the furthest candidate within max_dur, or the nearest one
    beyond it if there is none."""
    cuts, last, i = [], start, 0
    while end - last > max_dur:
        reach = [c for c in cands[i:] if c - last <= max_dur]
        if reach:
            c = reach[-1]
        else:
            later = [c for c in cands[i:] if c > last]
            if not later:
                break
            c = later[0]
        cuts.append(c)
        last = c
        i = cands.index(c) + 1
    return cuts


def safe_cut_candidates(
    segs: List[Segment], g_start: float, g_end: float
) -> Dict[float, str]:
    """Cut points where nobody else is speaking: time -> best cut type."""

    def covered(t: float, owner: Optional[Segment] = None) -> bool:
        return any(o is not owner and o.start < t < o.end for o in segs)

    cand: Dict[float, str] = {}
    starts = {o.start for o in segs}
    for o in segs:
        if o.end in starts and g_start < o.end < g_end and not covered(o.end):
            cand[o.end] = "turn"
    for o in segs:
        for t, c in o.boundaries.items():
            if CUT_LEVEL[c] < CUT_LEVEL.get(cand.get(t), 9) and not covered(t, o):
                cand[t] = c
    return cand


def split_group(segs: List[Segment], g_start: float, g_end: float, max_dur: float):
    """Split by length only. Returns list of (cut_time, cut_type)."""
    if max_dur <= 0 or g_end - g_start <= max_dur:
        return []

    cand = safe_cut_candidates(segs, g_start, g_end)
    for level in (1, 2, 3):
        pts = sorted(t for t, c in cand.items() if CUT_LEVEL[c] <= level)
        cuts = find_cuts(g_start, g_end, pts, max_dur)
        if cuts is not None:
            return [(t, cand[t]) for t in cuts]
    stats["groups_forced"] += 1
    pts = sorted(cand)
    return [(t, "forced") for t in greedy_cuts(g_start, g_end, pts, max_dur)]


def joint_split(
    segs: List[Segment],
    g_start: float,
    g_end: float,
    max_dur: float,
    max_spk: int,
    max_level: int,
):
    """Split by speaker count and length jointly.

    A piece is usable if it has <= max_spk speakers and (if max_dur > 0) lasts
    <= max_dur; unusable pieces are dropped later. Among safe cut points of
    level <= max_level, choose the cuts that maximise, in order: the number of
    words in usable pieces, -#cuts, -#word-level cuts, -#comma-level cuts.
    Returns list of (cut_time, cut_type).
    """
    # Sort key consistent with the word assignment in process_meeting: a word
    # goes to the piece containing its midpoint; a zero-duration word exactly
    # at a cut goes to the piece its segment starts in.
    words = sorted(
        ((w.start + w.end) / 2 + (1e-9 if s.start >= (w.start + w.end) / 2 else -1e-9),
         s.speaker)
        for s in segs
        for w in s.words
    )
    nspk = len({spk for _, spk in words})
    if nspk <= max_spk and (max_dur <= 0 or g_end - g_start <= max_dur):
        return []

    cand = {
        t: c for t, c in safe_cut_candidates(segs, g_start, g_end).items()
        if CUT_LEVEL[c] <= max_level
    }
    pts = [g_start] + sorted(cand) + [g_end]
    n = len(pts)
    keys = [k for k, _ in words]
    idx = [bisect.bisect_left(keys, p) for p in pts]
    idx[0], idx[-1] = 0, len(words)

    def extend(score, j, nwords):
        if j == n - 1:  # group end, not a cut
            return (score[0] + nwords,) + score[1:]
        lv = CUT_LEVEL[cand[pts[j]]]
        return (score[0] + nwords, score[1] - 1, score[2] - (lv == 3), score[3] - (lv == 2))

    # best[j] = (score, prev); prefix[j] = best (score, i) over i <= j, used for
    # dropped (zero-value) pieces, which may have any length.
    best = [None] * n
    best[0] = ((0, 0, 0, 0), -1)
    prefix = [best[0]] + [None] * (n - 1)
    for j in range(1, n):
        # drop the piece (pts[i], pts[j]) for the best earlier i
        best[j] = (extend(prefix[j - 1][0], j, 0), prefix[j - 1][1])
        # keep a usable piece (pts[i], pts[j]); grow it leftwards
        spk = set()
        for i in range(j - 1, -1, -1):
            if max_dur > 0 and pts[j] - pts[i] > max_dur:
                break
            spk.update(s for _, s in words[idx[i] : idx[i + 1]])
            if len(spk) > max_spk:
                break
            cand_score = extend(best[i][0], j, idx[j] - idx[i])
            if cand_score > best[j][0]:
                best[j] = (cand_score, i)
        prefix[j] = (best[j][0], j) if best[j][0] > prefix[j - 1][0] else prefix[j - 1]

    cuts, j = [], n - 1
    while best[j][1] > 0:
        j = best[j][1]
        cuts.append(pts[j])
    return [(t, cand[t]) for t in sorted(cuts)]


def normalize_word(w: Word) -> List[Tuple[float, float, str]]:
    """Kaldi-style AMI normalisation; a word may become several tokens (e.g.
    T_V_s -> T V S), in which case its duration is shared evenly."""
    toks = normalize_text_ami(w.text, normalize="kaldi").split()
    if not toks:
        stats["words_empty_after_norm"] += 1
        return []
    if len(toks) > 1:
        stats["words_split_by_norm"] += 1
    d = (w.end - w.start) / len(toks)
    return [(w.start + k * d, w.start + (k + 1) * d, t) for k, t in enumerate(toks)]


def drop_words_beyond(segs: List[Segment], reco_dur: float) -> List[Segment]:
    """Annotations of a few meetings (e.g. TS3007c) run past the end of the
    audio; drop those words."""
    out = []
    for s in segs:
        words = [w for w in s.words if w.end <= reco_dur]
        stats["words_beyond_audio_dropped"] += len(s.words) - len(words)
        if not words:
            continue
        if len(words) < len(s.words):
            end = max(w.end for w in words)
            s = Segment(
                s.speaker, s.start, end, words,
                {t: c for t, c in s.boundaries.items() if t < end},
            )
        out.append(s)
    return out


def process_meeting(meeting: str, segs: List[Segment], reco_dur: float, args):
    segs = sorted(drop_words_beyond(segs, reco_dur), key=lambda s: (s.start, s.end))

    # 1. group by overlap/touch
    groups = []
    for s in segs:
        if groups and s.start <= groups[-1]["end"]:
            groups[-1]["end"] = max(groups[-1]["end"], s.end)
            groups[-1]["segs"].append(s)
        else:
            groups.append({"start": s.start, "end": s.end, "segs": [s]})
    stats["groups_before_split"] += len(groups)
    if args.max_dur > 0:
        stats["groups_over_max_before_split"] += sum(
            1 for g in groups if g["end"] - g["start"] > args.max_dur
        )

    # 2. split long groups into pieces: (start, end, left_type, right_type, segs)
    pieces = []
    for gi, g in enumerate(groups):
        if args.max_speakers > 0:
            cuts = joint_split(
                g["segs"], g["start"], g["end"],
                args.max_dur, args.max_speakers, args.max_cut_level,
            )
        else:
            cuts = split_group(g["segs"], g["start"], g["end"], args.max_dur)
        for _, c in cuts:
            stats[f"cut_{c}"] += 1
        bounds = [(g["start"], "silence")] + cuts + [(g["end"], "silence")]
        for (a, ta), (b, tb) in zip(bounds, bounds[1:]):
            pieces.append(dict(start=a, end=b, left=ta, right=tb, group=gi, segs=g["segs"]))

    # 3. assign words to pieces, pad silence edges
    out = []
    for k, p in enumerate(pieces):
        a, b = p["start"], p["end"]
        seg_words = []  # (seg_start, speaker, [(start, end, token)])
        nspk = set()
        intervals = []
        left_cut, right_cut = p["left"] != "silence", p["right"] != "silence"
        for s in p["segs"]:
            if s.end < a or s.start > b:
                continue
            ws = []
            for w in s.words:
                # A word belongs to the piece containing its midpoint. Cuts never
                # fall strictly inside a word, so ties only happen for
                # zero-duration words exactly at a cut: they go to the piece
                # their segment starts in.
                mid = (w.start + w.end) / 2
                if mid < a or mid > b:
                    continue
                if left_cut and mid == a and s.start < a:
                    continue
                if right_cut and mid == b and s.start >= b:
                    continue
                ws.extend(normalize_word(w))
            if min(s.end, b) > max(s.start, a):
                intervals.append((max(s.start, a), min(s.end, b)))
            if ws:
                seg_words.append((max(s.start, a), s.speaker, ws))
                nspk.add(s.speaker)
        nwords = sum(len(x[2]) for x in seg_words)
        if nwords == 0:
            stats["pieces_without_words_dropped"] += 1
            continue
        if args.max_speakers > 0:
            if len(nspk) > args.max_speakers:
                stats["pieces_over_max_speakers_dropped"] += 1
                stats["words_in_dropped_pieces"] += nwords
                continue
            if args.max_dur > 0 and b - a > args.max_dur:
                stats["pieces_over_max_dur_dropped"] += 1
                stats["words_in_dropped_pieces"] += nwords
                continue
        stats["words_kept"] += nwords

        # padding (only on silence edges, at most half the gap)
        start, end = a, b
        if p["left"] == "silence":
            prev_end = pieces[k - 1]["end"] if k > 0 else 0.0
            start = max(0.0, a - min(args.pad, (a - prev_end) / 2))
        if p["right"] == "silence":
            next_start = pieces[k + 1]["start"] if k + 1 < len(pieces) else reco_dur
            end = min(reco_dur, b + min(args.pad, max(0.0, next_start - b) / 2))
        # Round first so that `segments`, the group id and the ctm agree.
        start, end = round(start, 2), round(end, 2)

        # max number of concurrently active segments (strict overlap)
        events = sorted([(x, 1) for x, _ in intervals] + [(y, -1) for _, y in intervals],
                        key=lambda e: (e[0], e[1]))
        active = max_conc = 0
        for _, d in events:
            active += d
            max_conc = max(max_conc, active)

        gid = f"AMI_{meeting}_{round(start * 100):07d}_{round(end * 100):07d}"
        seg_words.sort(key=lambda x: (x[0], x[1]))
        text = " ".join(t for _, _, ws in seg_words for _, _, t in ws)
        ctm = sorted(
            (round(ws_s - start, 3), spk, round(ws_e - ws_s, 3), tok)
            for _, spk, ws in seg_words
            for ws_s, ws_e, tok in ws
        )
        out.append(
            dict(
                gid=gid, meeting=meeting, start=start, end=end, text=text, ctm=ctm,
                info=(round(end - start, 2), len(nspk), max_conc, p["left"], p["right"], nwords),
            )
        )
    return out


def write_split(out_dir: Path, utts: List[dict], wavs: Dict[str, Path], durs: Dict[str, float]):
    out_dir.mkdir(parents=True, exist_ok=True)
    utts = sorted(utts, key=lambda u: u["gid"])
    ids = [u["gid"] for u in utts]
    assert len(ids) == len(set(ids)), "duplicate group ids"
    meetings = sorted({u["meeting"] for u in utts})
    with open(out_dir / "wav.scp", "w") as f:
        for m in meetings:
            f.write(f"{m} {wavs[m]}\n")
    with open(out_dir / "reco2dur", "w") as f:
        for m in meetings:
            f.write(f"{m} {durs[m]:.3f}\n")
    with open(out_dir / "segments", "w") as f:
        for u in utts:
            f.write(f"{u['gid']} {u['meeting']} {u['start']:.2f} {u['end']:.2f}\n")
    with open(out_dir / "text", "w") as f:
        for u in utts:
            f.write(f"{u['gid']} {u['text']}\n")
    with open(out_dir / "utt2spk", "w") as f:
        for u in utts:
            f.write(f"{u['gid']} {u['meeting']}\n")
    spk2utt = defaultdict(list)
    for u in utts:
        spk2utt[u["meeting"]].append(u["gid"])
    with open(out_dir / "spk2utt", "w") as f:
        for m in meetings:
            f.write(f"{m} {' '.join(spk2utt[m])}\n")
    with open(out_dir / "words.ctm", "w") as f:
        for u in utts:
            for st, spk, dur, tok in u["ctm"]:
                f.write(f"{u['gid']} {spk} {st:.3f} {dur:.3f} {tok}\n")
    with open(out_dir / "utt2info", "w") as f:
        for u in utts:
            f.write(f"{u['gid']} {' '.join(str(x) for x in u['info'])}\n")
    tot = sum(u["end"] - u["start"] for u in utts)
    logging.info(
        f"{out_dir}: {len(meetings)} meetings, {len(utts)} groups, {tot / 3600:.2f} h"
    )


def main():
    args = get_args()
    spk_map = load_speaker_map(args.annotations_dir)
    meetings = load_segments(args.annotations_dir, spk_map)
    logging.info(f"Loaded segments for {len(meetings)} meetings")

    partition = PARTITIONS[args.partition]
    meeting2split = {m: s for s, ms in partition.items() for m in ms}
    for m in sorted(set(meetings) - set(meeting2split)):
        logging.warning(f"{m} is not in partition {args.partition}, skipped")
    for m in sorted(set(meeting2split) - set(meetings)):
        logging.warning(f"{m} is in partition {args.partition} but has no annotations")

    wavs, durs = {}, {}
    split_utts = defaultdict(list)
    for m in sorted(meetings):
        if m not in meeting2split or meeting2split[m] not in args.splits:
            continue
        wav = (args.audio_dir / args.wav_template.format(meeting=m)).resolve()
        if not wav.exists():
            raise FileNotFoundError(wav)
        wavs[m] = wav
        durs[m] = sf.info(str(wav)).duration
        split_utts[meeting2split[m]].extend(process_meeting(m, meetings[m], durs[m], args))

    for split, utts in split_utts.items():
        write_split(args.out_dir / split, utts, wavs, durs)
    for k, v in sorted(stats.items()):
        logging.info(f"  {k}: {v}")


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
