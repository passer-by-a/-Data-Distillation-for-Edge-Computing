"""汇总主表（Markdown + CSV），按方法聚合多个随机种子"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDER = ["frozen_stream_dm", "online_naive_dm", "online_replay_dm",
         "online_replay_align_dm", "online_moment", "online_kmeans",
         "online_herding", "online_reservoir", "full_method"]


def load_all(results_dir):
    out = []
    for p in glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True):
        if "_stale_pre_audit" in p.split(os.sep):
            continue
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if "summary" not in o:
            continue
        if "rounds" not in o and o["summary"].get("scenario") != "offline_non_causal":
            continue
        if o["summary"].get("partial"):
            continue
        o["_path"] = p
        out.append(o)
    return out


def _ms(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def _fmt(ms, nd=2):
    if ms is None:
        return "—"
    return f"{ms[0]:.{nd}f}±{ms[1]:.{nd}f}"


def aggregate(objs, scenario, methods=None, append_extra=True):
    groups = defaultdict(list)
    for o in objs:
        s = o["summary"]
        if s.get("scenario") == scenario:
            groups[s["method"]].append(o)
    order = list(methods) if methods is not None else ORDER
    if append_extra:
        order = order + [k for k in groups if k not in order]
    rows = []
    for m in order:
        runs = groups.get(m)
        if not runs:
            continue
        ss = [r["summary"] for r in runs]
        lasts = [r["rounds"][-1] for r in runs]
        row = {
            "method": m, "seeds": len(runs),
            "n_reps": runs[0]["summary"].get("n_reps"),
            "initial": _ms([s["initial_acc"] for s in ss]),
            "final": _ms([s["final_acc"] for s in ss]),
            "online": _ms([s["avg_online_acc"] for s in ss]),
            "f1": _ms([s["final_f1"] for s in ss]),
            "int8": _ms([s.get("final_test_int8_acc") for s in ss]),
            "comm_kb": float(np.mean([s["cumulative_upload_bytes"] for s in ss]) / 1024),
            "sync": runs[0]["summary"]["sync_mode"],
            "var_ratio": _ms([s.get("final_group_var_ratio") for s in ss]),
            "align_err": _ms([s.get("final_alignment_err") for s in ss]),
            "n_seen": _ms([s.get("stats_seen_final") for s in ss]),
            "edge_sec": _ms([s.get("edge_total_sec") for s in ss]),
            "cloud_sec": _ms([s.get("cloud_total_sec") for s in ss]),
            "peak_gb": _ms([s.get("peak_gpu_gb") for s in ss]),
        }
        if scenario == "label_shift":
            for g in ("current", "historical"):
                row[g] = _ms([x["groups"].get(g, {}).get("acc")
                              for x in lasts])
        if scenario == "covariate_shift":
            row["clean"] = _ms([x["groups"].get("historical_domain", {}).get("acc")
                                for x in lasts])
            row["shifted"] = _ms([x["groups"].get("current_domain", {}).get("acc")
                                  for x in lasts])
        rows.append(row)
    return rows


def write_markdown(objs, out_path, dataset="cifar100"):
    lines = [f"# {dataset} Streaming 结果汇总", "",
             "> 5 轮，Initial Base 20%，代表预算 980（最终 50:1），INT8。"
             " 数值为均值±标准差（种子 42/43/44）。", ""]
    iid = aggregate(objs, "iid", methods=ORDER, append_extra=False)
    lines += ["## 1. IID", "",
              "| 方法 | 种子 | 初始 | 最终 | 在线平均 | 最终 F1 | INT8 | 累计通信 | 方差比 | 对齐误差 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in iid:
        lines.append(
            f"| {r['method']} | {r['seeds']} | {_fmt(r['initial'])} | "
            f"{_fmt(r['final'])} | {_fmt(r['online'])} | {_fmt(r['f1'])} | "
            f"{_fmt(r['int8'])} | {r['comm_kb']:.0f} KB | "
            f"{_fmt(r['var_ratio'], 3)} | {_fmt(r['align_err'], 3)} |")
    ab = [r for r in aggregate(objs, "iid") if r["method"] not in ORDER]
    if ab:
        lines += ["", "## 1b. 消融（单种子 42）", "",
                  "| 方法 | 最终 | 在线平均 | 累计通信 | 预算 |",
                  "|---|---|---|---|---|"]
        for r in ab:
            lines.append(f"| {r['method']} | {_fmt(r['final'])} | "
                         f"{_fmt(r['online'])} | {r['comm_kb']:.0f} KB | "
                         f"{r['n_reps']} |")
    ls = aggregate(objs, "label_shift")
    lines += ["", "## 2. Label Shift", "",
              "| 方法 | 最终 | 当前类 | 历史类 | 累计通信 |", "|---|---|---|---|---|"]
    for r in ls:
        lines.append(f"| {r['method']} | {_fmt(r['final'])} | {_fmt(r.get('current'))} | "
                     f"{_fmt(r.get('historical'))} | {r['comm_kb']:.0f} KB |")
    cs = aggregate(objs, "covariate_shift")
    lines += ["", "## 3. Covariate Shift", "",
              "| 方法 | 原域 | 当前域 | 累计通信 |", "|---|---|---|---|"]
    for r in cs:
        lines.append(f"| {r['method']} | {_fmt(r.get('clean'))} | "
                     f"{_fmt(r.get('shifted'))} | {r['comm_kb']:.0f} KB |")
    off = [o for o in objs if o["summary"].get("scenario") == "offline_non_causal"]
    if off:
        lines += ["", "## 4. 非因果离线参考（全量 49k 一次性 Batch DM）", "",
                  "| 参考 | 方法 | 编码器可访问数据 | 归一化 | 最终 |",
                  "|---|---|---|---|---|"]
        for o in sorted(off, key=lambda x: x["summary"].get(
                "tag", x["summary"]["method"])):
            s = o["summary"]
            lines.append(f"| {s.get('tag', s['method'])} | {s['method']} | "
                         f"{s.get('vae_accessible')} | "
                         f"{s.get('protocol_source')} | {s['final_acc']:.2f} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="vae_streaming/results")
    ap.add_argument("--dataset", default=None,
                    help="cifar10 / cifar100；缺省则汇总全部")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    objs = load_all(os.path.join(ROOT, args.results))
    if args.dataset:
        objs = [o for o in objs
                if o["summary"].get("dataset") == args.dataset]
        out = args.out or f"vae_streaming/results/{args.dataset}/main_table.md"
    else:
        out = args.out or "vae_streaming/results/main_table.md"
    p = os.path.join(ROOT, out)
    write_markdown(objs, p, dataset=args.dataset or "CIFAR-100/10")
    print(open(p).read())


if __name__ == "__main__":
    main()
