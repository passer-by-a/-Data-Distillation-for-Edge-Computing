"""汇总 80 轮多分支在线配对实验

输入：
  vae_streaming/results/cifar10/iid/{method}_multibranch80_seed{seed}.json

输出：
  multibranch80_summary.json  逐分支末轮/在线/分段配对差与决策
  multibranch80_curves.json   每轮 val/test/INT8 准确率与 NLL

用法：
  python -m vae_streaming.experiments.summarize_multibranch
"""
import argparse
import json
import os

import numpy as np

from ..config_utils import load_config, result_path

BRANCHES = ("inc200", "inc20", "scratch200", "scratch20")
SEGMENTS = ((0, 20), (20, 40), (40, 60), (60, 80))


def _stream_rounds(doc):
    return [r for r in doc["rounds"] if int(r["round"]) >= 0]


def _series(doc, branch):
    rl = _stream_rounds(doc)
    out = {"round": [int(r["round"]) for r in rl]}
    for split in ("val", "test", "test_int8"):
        for key in ("acc", "f1", "nll"):
            vals = []
            for r in rl:
                m = r["cloud_branches"][branch].get(split)
                vals.append(None if m is None else float(m[key]))
            out[f"{split}_{key}"] = vals
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _std(xs):
    xs = [x for x in xs if x is not None]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def _seg_mean(diff, lo, hi):
    return float(np.mean(diff[lo:hi])) if len(diff) >= hi else None


def _cost(doc, branch):
    rl = _stream_rounds(doc)
    wall = [float(r["cloud"]["branch_wall_sec"][branch]) for r in rl
            if r.get("cloud", {}).get("branch_wall_sec")]
    steps = [int(r["cloud"]["steps"][branch]) for r in rl
             if r.get("cloud", {}).get("steps")]
    return {"mean_wall_sec": _mean(wall),
            "total_wall_sec": float(np.sum(wall)) if wall else None,
            "steps_per_round": int(steps[0]) if steps else None,
            "total_steps": int(np.sum(steps)) if steps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="vae_streaming/configs/cifar10_multi_branch80.yaml")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--methods", default="dm,online_reservoir")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--tag", default="multibranch80")
    ap.add_argument("--candidate", default="scratch20")
    ap.add_argument("--baseline", default="inc200")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    summary = {"tag": args.tag, "seeds": seeds,
               "candidate": args.candidate, "baseline": args.baseline,
               "methods": {}}
    curves = {}

    for method in methods:
        per_seed = {}
        for seed in seeds:
            p = result_path(
                cfg, f"{args.dataset}/iid/{method}_{args.tag}_seed{seed}.json")
            with open(p, "r", encoding="utf-8") as f:
                doc = json.load(f)
            base = doc["rounds"][0]
            base_branch = base["cloud_branches"]["inc200"]
            curves.setdefault(method, {})[str(seed)] = {
                b: _series(doc, b) for b in BRANCHES}
            per_seed[seed] = {
                "base": {"val": base_branch["val"],
                         "test": base_branch["test"]},
                "branches": {},
                "cost": {b: _cost(doc, b) for b in BRANCHES},
            }
            for b in BRANCHES:
                s = _series(doc, b)
                per_seed[seed]["branches"][b] = {
                    "final_val": s["val_acc"][-1],
                    "final_test": s["test_acc"][-1],
                    "final_test_int8": s["test_int8_acc"][-1],
                    "final_val_nll": s["val_nll"][-1],
                    "final_test_nll": s["test_nll"][-1],
                    "online_val": _mean(s["val_acc"]),
                    "online_test": _mean(s["test_acc"]),
                    "online_test_int8": _mean(s["test_int8_acc"]),
                    "online_val_nll": _mean(s["val_nll"]),
                    "online_test_nll": _mean(s["test_nll"]),
                }

        # 配对差（同一 seed、同一轮）
        paired = {}
        for b in ("inc20", "scratch200", "scratch20"):
            diffs = {}
            for seed in seeds:
                s_b = curves[method][str(seed)][b]
                s_0 = curves[method][str(seed)][args.baseline]
                d = {"val": [], "test": [], "test_int8": [],
                     "val_nll": [], "test_nll": []}
                for k in d:
                    for x, y in zip(s_b[f"{k}_acc"] if "nll" not in k
                                    else s_b[k],
                                    s_0[f"{k}_acc"] if "nll" not in k
                                    else s_0[k]):
                        d[k].append(None if x is None or y is None
                                    else float(x - y))
                diffs[seed] = d
            paired[b] = {
                "final_val_diff": [diffs[s]["val"][-1] for s in seeds],
                "final_test_diff": [diffs[s]["test"][-1] for s in seeds],
                "online_val_diff": [_mean(diffs[s]["val"]) for s in seeds],
                "online_test_diff": [_mean(diffs[s]["test"]) for s in seeds],
                "final_val_nll_diff": [diffs[s]["val_nll"][-1] for s in seeds],
                "final_test_nll_diff": [diffs[s]["test_nll"][-1]
                                        for s in seeds],
                "segments_val": {
                    f"r{lo}-{hi - 1}": [_seg_mean(diffs[s]["val"], lo, hi)
                                        for s in seeds]
                    for lo, hi in SEGMENTS},
                "segments_test": {
                    f"r{lo}-{hi - 1}": [_seg_mean(diffs[s]["test"], lo, hi)
                                        for s in seeds]
                    for lo, hi in SEGMENTS},
            }
            paired[b]["mean_final_val_diff"] = _mean(
                paired[b]["final_val_diff"])
            paired[b]["mean_final_test_diff"] = _mean(
                paired[b]["final_test_diff"])
            paired[b]["mean_online_val_diff"] = _mean(
                paired[b]["online_val_diff"])
            paired[b]["mean_online_test_diff"] = _mean(
                paired[b]["online_test_diff"])

        branch_stats = {}
        for b in BRANCHES:
            branch_stats[b] = {
                "final_val": [_mean([per_seed[s]["branches"][b]["final_val"]])
                              for s in seeds],
                "final_test": [per_seed[s]["branches"][b]["final_test"]
                               for s in seeds],
                "online_val": [per_seed[s]["branches"][b]["online_val"]
                               for s in seeds],
                "online_test": [per_seed[s]["branches"][b]["online_test"]
                                for s in seeds],
                "final_val_mean": _mean([per_seed[s]["branches"][b]
                                         ["final_val"] for s in seeds]),
                "final_test_mean": _mean([per_seed[s]["branches"][b]
                                          ["final_test"] for s in seeds]),
                "online_val_mean": _mean([per_seed[s]["branches"][b]
                                          ["online_val"] for s in seeds]),
                "online_test_mean": _mean([per_seed[s]["branches"][b]
                                           ["online_test"] for s in seeds]),
                "final_val_std": _std([per_seed[s]["branches"][b]
                                       ["final_val"] for s in seeds]),
                "final_test_std": _std([per_seed[s]["branches"][b]
                                        ["final_test"] for s in seeds]),
                "cost": {k: _mean([per_seed[s]["cost"][b][k] for s in seeds])
                         for k in ("mean_wall_sec", "total_wall_sec")},
                "steps_per_round": per_seed[seeds[0]]["cost"][b]
                                   ["steps_per_round"],
            }

        cand = paired[args.candidate]
        decision = {
            "candidate": args.candidate,
            "mean_final_val_diff": cand["mean_final_val_diff"],
            "final_val_diff_per_seed": cand["final_val_diff"],
            "pass_final": bool(cand["mean_final_val_diff"] is not None
                               and cand["mean_final_val_diff"] >= 0.5),
            "mean_online_val_diff": cand["mean_online_val_diff"],
            "online_val_diff_per_seed": cand["online_val_diff"],
            "pass_online": bool(cand["mean_online_val_diff"] is not None
                                and cand["mean_online_val_diff"] >= 0.0),
            "all_seeds_final_positive": bool(all(
                x is not None and x > 0 for x in cand["final_val_diff"])),
            "early20_val_diff": cand["segments_val"]["r0-19"],
            "segments_val": cand["segments_val"],
            "segments_test": cand["segments_test"],
            "candidate_steps_per_round": branch_stats[args.candidate]
                                         ["steps_per_round"],
            "baseline_steps_per_round": branch_stats[args.baseline]
                                        ["steps_per_round"],
            "step_saving": (1.0 - branch_stats[args.candidate]
                            ["steps_per_round"] /
                            branch_stats[args.baseline]["steps_per_round"]),
            "candidate_mean_wall_sec": branch_stats[args.candidate]["cost"]
                                       ["mean_wall_sec"],
            "baseline_mean_wall_sec": branch_stats[args.baseline]["cost"]
                                      ["mean_wall_sec"],
        }
        summary["methods"][method] = {
            "base": {s: per_seed[s]["base"] for s in seeds},
            "branch_stats": branch_stats,
            "paired_vs_baseline": paired,
            "decision": decision,
        }

    out = args.out or result_path(
        cfg, f"{args.dataset}/iid/{args.tag}_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    cpath = result_path(
        cfg, f"{args.dataset}/iid/{args.tag}_curves.json")
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(curves, f, ensure_ascii=False, indent=2)

    for method in methods:
        m = summary["methods"][method]
        print(f"\n===== {method} =====")
        print("base:", json.dumps(m["base"][seeds[0]]["test"], ensure_ascii=False))
        print(f"{'branch':>10} {'final_val':>18} {'final_test':>18} "
              f"{'online_val':>8} {'online_test':>8} {'steps':>6} {'wall_s':>7}")
        for b in BRANCHES:
            st = m["branch_stats"][b]
            print(f"{b:>10} {st['final_val_mean']:>8.2f}±{st['final_val_std']:<7.2f} "
                  f"{st['final_test_mean']:>8.2f}±{st['final_test_std']:<7.2f} "
                  f"{st['online_val_mean']:>8.2f} {st['online_test_mean']:>8.2f} "
                  f"{st['steps_per_round']:>6} {st['cost']['mean_wall_sec']:>7.3f}")
        d = m["decision"]
        print("decision:", json.dumps({
            "candidate": d["candidate"],
            "final_val_diff": d["final_val_diff_per_seed"],
            "mean_final_val_diff": d["mean_final_val_diff"],
            "online_val_diff": d["online_val_diff_per_seed"],
            "mean_online_val_diff": d["mean_online_val_diff"],
            "pass_final": d["pass_final"], "pass_online": d["pass_online"],
            "early20_val_diff": d["early20_val_diff"],
            "step_saving": d["step_saving"],
            "wall_saving": (1 - d["candidate_mean_wall_sec"] /
                            d["baseline_mean_wall_sec"]),
        }, ensure_ascii=False, indent=2))
    print(f"-> {out}\n-> {cpath}")


if __name__ == "__main__":
    main()
