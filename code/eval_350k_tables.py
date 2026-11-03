"""Numbers for the 350k report tables: per model x checkpoint success by evaluation type,
per-task wins, held-out examples. usage: python code/eval_350k_tables.py"""

from __future__ import annotations

from collections import defaultdict

from eval_350k_figs import CKPTS, MODELS, eps, sr, task


def main() -> None:
    names = list(MODELS)
    print("| model | ckpt | train scenes | new scenes (n) | held-out sets (n) | held-out tasks | total |")
    data = {}
    for c in CKPTS:
        for m in names:
            s = MODELS[m]
            tr, nw, held = eps(s["train"](c)), eps(s["new"].format(c)), eps(s["held"].format(c))
            hs = [r for r in held if r["group"] == "heldout_sets"]
            ht = [r for r in held if r["group"] == "heldout_tasks"]
            data[(m, c)] = {"train": tr, "new": nw, "hs": hs, "ht": ht}
            print(f"| {m} | {c} | {sr(tr):.1%} ({len(tr)}) | {sr(nw):.1%} ({len(nw)}) | "
                  f"{sr(hs):.1%} ({len(hs)}) | {sum(r['success'] for r in ht)}/{len(ht)} | {sr(nw + hs + ht):.1%} |")
    for c in CKPTS:
        for a, b in ((names[0], names[2]), (names[0], names[1])):
            ta = defaultdict(list)
            tb = defaultdict(list)
            for r in data[(a, c)]["new"]:
                ta[task(r)].append(r["success"])
            for r in data[(b, c)]["new"]:
                tb[task(r)].append(r["success"])
            common = [t for t in ta if t in tb and len(ta[t]) == len(tb[t])]
            w = sum(sum(ta[t]) > sum(tb[t]) for t in common)
            e = sum(sum(ta[t]) == sum(tb[t]) for t in common)
            print(f"{c} per task new scenes {a} vs {b}: better {w} / equal {e} / worse {len(common) - w - e} (of {len(common)})")
    for cs in ("place_a2b_bp/I840/L0/A_left_of_B", "place_shoe_bp/I1/L1/toe_left"):
        print(cs, {m: f"{sum(r['success'] for r in data[(m, 'c80k')]['hs'] if r['candidate_set'] == cs)}/"
                      f"{sum(1 for r in data[(m, 'c80k')]['hs'] if r['candidate_set'] == cs)}" for m in names})


if __name__ == "__main__":
    main()
