#!/usr/bin/env python3
"""
Break down t-SOT cpWER errors on 2-speaker groups by where the reference words
are in time, using the reference word timings / speakers of the manifest.

Per group, cpWER (MeetEval) picks the speaker <-> channel assignment; each
assigned pair is aligned word by word (Levenshtein, same error count as
MeetEval), so every reference word is correct / substituted / deleted, and
each insertion is attributed to the preceding reference word of that speaker
(the following one at the start). An unassigned speaker counts as deletions
of all its words; an unassigned hypothesis channel as insertions, reported
separately ("extra channel").

Reference word categories (by the word's [start, end] vs the other speaker's
words in the group):
  overlap      the other speaker is speaking at the same time
  near_change  no overlap, but the other speaker speaks within --near seconds
  single       the other speaker is more than --near seconds away
and by speaker role: major (more words in the group) / minor.

Usage:
  python local/analyze_tsot_errors.py --exp-dirs exp/cfg2 exp/cfg8 exp/cfg16 \
      --latencies full 320ms
"""
import argparse
import glob
import json
from collections import defaultdict

import meeteval.wer as mw
from lhotse import load_manifest_lazy

LAT = {"320ms": "chunk-16_left-context-128", "640ms": "chunk-32_left-context-256",
       "1280ms": "chunk-64_left-context-256", "full": "chunk--1_left-context--1"}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dirs", nargs="+", required=True)
    p.add_argument("--latencies", nargs="+", default=["full", "320ms"])
    p.add_argument("--set", default="test_2spk")
    p.add_argument("--near", type=float, default=1.0)
    p.add_argument("--manifest",
                   default="data/manifests/ami_ihm_mix_utterance_group_supervisions_test.jsonl.gz")
    return p.parse_args()


def align(ref, hyp):
    """Levenshtein alignment. Returns per-ref-word op in {C,S,D} and the number
    of insertions attached to each ref position (-1: before the first word)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                          d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops = [None] * n
    ins = defaultdict(int)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops[i - 1] = "C" if ref[i - 1] == hyp[j - 1] else "S"
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops[i - 1] = "D"
            i -= 1
        else:
            ins[i - 1] += 1  # inserted after ref word i-1
            j -= 1
    return ops, ins, d[n][m]


def categorize(sup, near):
    """{speaker: [(word, category, role), ...]} in time order."""
    words = sorted(zip(sup.alignment["word"], sup.custom["word_speakers"]), key=lambda x: x[0][1])
    per = defaultdict(list)
    for w, spk in words:
        per[spk].append((w[0], w[1], w[1] + w[2]))
    spks = list(per)
    assert len(spks) == 2, sup.id
    major = max(spks, key=lambda s: len(per[s]))
    out = {}
    for s in spks:
        other = per[spks[1] if s == spks[0] else spks[0]]
        cats = []
        for w, b, e in per[s]:
            if any(ob < e and b < oe for _, ob, oe in other):
                c = "overlap"
            else:
                gap = min(max(ob - e, b - oe) for _, ob, oe in other)
                c = "near_change" if gap < near else "single"
            cats.append((w, c, "major" if s == major else "minor"))
        out[s] = cats
    return out


def analyze(path, sups, near):
    # stats[key] = [ref words, S, D, I]
    stats = defaultdict(lambda: [0, 0, 0, 0])
    extra_ins = 0
    total_err = total_ref = 0
    for line in open(path):
        r = json.loads(line)
        cat = categorize(sups[r["id"]], near)
        spk_keys = sorted(r["ref_by_speaker"])
        refs = [r["ref_by_speaker"][k] for k in spk_keys]
        hyps = [c for c in r["hyp_channels"] if c.strip()]
        er = mw.cp_word_error_rate(refs, hyps, reference_sort=False, hypothesis_sort=False)
        assigned = {ri: hi for ri, hi in er.assignment if ri is not None and hi is not None}
        used_h = set(assigned.values())
        g_err = 0
        for ri, k in enumerate(spk_keys):
            items = cat[k]
            assert [w for w, _, _ in items] == refs[ri].split(), r["id"]
            if ri in assigned:
                ops, ins, e = align(refs[ri].split(), hyps[assigned[ri]].split())
            else:
                ops, ins, e = ["D"] * len(items), {}, len(items)
            g_err += e
            for idx, ((_, c, role), op) in enumerate(zip(items, ops)):
                for key in (c, role, f"{role}/{c}", "all"):
                    st = stats[key]
                    st[0] += 1
                    st[1] += op == "S"
                    st[2] += op == "D"
            for pos, n in ins.items():
                _, c, role = items[max(pos, 0)]
                for key in (c, role, f"{role}/{c}", "all"):
                    stats[key][3] += n
        for hi, h in enumerate(hyps):
            if hi not in used_h:
                extra_ins += len(h.split())
                g_err += len(h.split())
        assert g_err == er.errors, (r["id"], g_err, er.errors)
        total_err += er.errors
        total_ref += er.length
    return stats, extra_ins, total_err, total_ref


def main():
    args = get_args()
    sups = {s.id: s for s in load_manifest_lazy(args.manifest)}
    for exp in args.exp_dirs:
        for lat in args.latencies:
            path = glob.glob(f"{exp}/*/hyps-{args.set}-modified_beam_search*epoch-50_avg-10_{LAT[lat]}_*.jsonl")
            assert len(path) == 1, (exp, lat, path)
            stats, extra, terr, tref = analyze(path[0], sups, args.near)
            print(f"== {exp} {lat}: cpWER {100 * terr / tref:.2f} ({terr} errors / {tref} words; "
                  f"extra-channel insertions {extra} = {100 * extra / terr:.1f}% of errors)")
            print(f"   {'category':20s} {'words':>7s} {'%words':>7s} {'WER':>6s} {'sub':>6s} {'del':>6s} {'ins':>6s} {'%errors':>8s}")
            for key in ["overlap", "near_change", "single", "major", "minor",
                        "major/overlap", "major/near_change", "major/single",
                        "minor/overlap", "minor/near_change", "minor/single"]:
                n, s, d, i = stats[key]
                if n == 0:
                    continue
                print(f"   {key:20s} {n:7d} {100 * n / tref:6.1f}% {100 * (s + d + i) / n:6.2f} "
                      f"{100 * s / n:6.2f} {100 * d / n:6.2f} {100 * i / n:6.2f} {100 * (s + d + i) / terr:7.1f}%")


if __name__ == "__main__":
    main()
