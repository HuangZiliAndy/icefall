#!/usr/bin/env python3
"""
Score t-SOT decoding results (hyps-*.jsonl written by zipformer/decode.py)
with cpWER (concatenated minimum-permutation WER, MeetEval).

Per utterance group:
  reference streams  = one per speaker (ref_by_speaker: all words of the
                       speaker in the group, in time order)
  hypothesis streams = the non-empty virtual channels of the t-SOT output
                       (hyp_channels; channel switches at every <cc>)
cpWER picks the speaker<->channel assignment with the fewest errors; words in
an unassigned channel count as insertions, an unassigned speaker as
deletions. Errors and reference words are summed over all groups (not an
average of per-group rates).

Also reported:
  spk_cnt_err  % of groups whose number of non-empty channels differs from the
               number of reference speakers
  cc_ratio     #<cc> in hypotheses / #<cc> in references (2-speaker sets;
               < 1: too few speaker changes predicted)
  cc_groups    % of groups with at least one <cc> in the hypothesis
               (1-speaker sets: spurious speaker changes)

Usage:
  python local/score_cpwer.py --exp-dirs exp/cfg2 exp/cfg3 \
      --sets test_1spk test_2spk --epoch 50 --avg 10 --out exp/cpwer_test.tsv
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import meeteval.wer as mw

LATENCY = {("16", "128"): "320ms", ("32", "256"): "640ms",
           ("64", "256"): "1280ms", ("-1", "-1"): "full"}
METHOD_SHORT = {"greedy_search": "greedy", "modified_beam_search": "mod_beam",
                "fast_beam_search": "fast_beam"}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--sets", type=str, nargs="+", default=["test_1spk", "test_2spk"])
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--avg", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def score_file(path: Path):
    ers = []
    n_groups = spk_cnt_err = cc_groups = cc_hyp = cc_ref = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            refs = [v for _, v in sorted(r["ref_by_speaker"].items())]
            hyps = [c for c in r["hyp_channels"] if c.strip()]
            ers.append(
                mw.cp_word_error_rate(
                    refs, hyps, reference_sort=False, hypothesis_sort=False
                )
            )
            n_groups += 1
            spk_cnt_err += len(hyps) != len(refs)
            n_cc = r["hyp"].split().count("<cc>")
            cc_hyp += n_cc
            cc_groups += n_cc > 0
            cc_ref += r["ref"].split().count("<cc>")
    return ers, dict(n_groups=n_groups, spk_cnt_err=spk_cnt_err,
                     cc_groups=cc_groups, cc_hyp=cc_hyp, cc_ref=cc_ref)


def summarize(ers, st):
    tot = mw.combine_error_rates(*ers)
    L = tot.length
    return {
        "cpWER": 100 * tot.errors / L,
        "ins": 100 * tot.insertions / L,
        "del": 100 * tot.deletions / L,
        "sub": 100 * tot.substitutions / L,
        "ref_words": L,
        "groups": st["n_groups"],
        "spk_cnt_err": 100 * st["spk_cnt_err"] / st["n_groups"],
        "cc_ratio": st["cc_hyp"] / st["cc_ref"] if st["cc_ref"] else float("nan"),
        "cc_groups": 100 * st["cc_groups"] / st["n_groups"],
    }


def main():
    args = get_args()
    # results[(exp, method, latency)][set] = (ers, stats)
    results = defaultdict(dict)
    pat = re.compile(
        rf"hyps-(?P<set>[^-]+)-(?P<method>[a-z_]+?)(_beam-size-\d+|_beam-[\d.]+_max-contexts-\d+_max-states-\d+)?"
        rf"-epoch-{args.epoch}_avg-{args.avg}_chunk-(?P<chunk>-?\d+)_left-context-(?P<left>-?\d+)_"
    )
    for exp in args.exp_dirs:
        for path in sorted(exp.glob(f"*/hyps-*-epoch-{args.epoch}_avg-{args.avg}_*.jsonl")):
            m = pat.match(path.name)
            if not m or m["set"] not in args.sets:
                continue
            lat = LATENCY.get((m["chunk"], m["left"]), f"c{m['chunk']}_l{m['left']}")
            key = (exp.name, METHOD_SHORT.get(m["method"], m["method"]), lat)
            assert m["set"] not in results[key], f"duplicate result for {key} {path}"
            results[key][m["set"]] = score_file(path)

    rows = []
    for key in sorted(results, key=lambda k: (k[0], list(METHOD_SHORT.values()).index(k[1])
                                               if k[1] in METHOD_SHORT.values() else 9,
                                               list(LATENCY.values()).index(k[2])
                                               if k[2] in LATENCY.values() else 9)):
        per_set = results[key]
        for s in args.sets:
            if s in per_set:
                rows.append((*key, s, summarize(*per_set[s])))
        if all(s in per_set for s in args.sets) and len(args.sets) > 1:
            ers = [e for s in args.sets for e in per_set[s][0]]
            st = defaultdict(int)
            for s in args.sets:
                for k, v in per_set[s][1].items():
                    st[k] += v
            rows.append((*key, "+".join(args.sets), summarize(ers, st)))

    cols = ["cpWER", "sub", "del", "ins", "spk_cnt_err", "cc_ratio", "cc_groups", "groups", "ref_words"]
    header = "\t".join(["exp", "method", "latency", "set"] + cols)
    lines = [header] + [
        "\t".join([*r[:4]] + [f"{r[4][c]:.2f}" if isinstance(r[4][c], float) else str(r[4][c]) for c in cols])
        for r in rows
    ]
    print("\n".join(lines))
    if args.out:
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
