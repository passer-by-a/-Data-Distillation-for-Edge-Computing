"""汇总主表（Markdown + CSV），按方法聚合多个随机种子"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORDER = ["dm", "dm_multi", "online_kmeans", "online_moment",
         "online_reservoir", "online_reservoir_latent", "kip", "exstream",
         "gcr"]


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
    # 与 summarize_rerun80 一致：多种子用样本标准差(ddof=1)
    ddof = 1 if len(vals) > 1 else 0
    return float(np.mean(vals)), float(np.std(vals, ddof=ddof))


def _fmt(ms, nd=2):
    if ms is None:
        return "—"
    return f"{ms[0]:.{nd}f}±{ms[1]:.{nd}f}"


def _representation(o):
    s = o["summary"]
    return str(s.get("representation")
               or (o.get("config") or {}).get("representation")
               or "vae")


def _seed_of(o):
    return int(o["summary"].get("seed", 0) or 0)


def aggregate(objs, scenario, methods=None, append_extra=True):
    groups = defaultdict(list)
    seen = {}
    for o in objs:
        s = o["summary"]
        if s.get("scenario") != scenario:
            continue
        rep = _representation(o)
        key = (s["method"], rep, _seed_of(o), s.get("num_rounds"))
        if key in seen:
            raise ValueError(
                "检测到重复结果（同一数据集/方法/表示/seed/轮数）：\n"
                f"  {seen[key]}\n  {o['_path']}")
        seen[key] = o["_path"]
        groups[(s["method"], rep)].append(o)
    order = list(methods) if methods is not None else ORDER
    if append_extra:
        extra = [k for k in groups if k[0] not in order]
        order = [(m, None) for m in order] + extra
    else:
        order = [(m, None) for m in order]
    rows = []
    for m, rep_filter in order:
        keys = [(km, kr) for (km, kr) in groups
                if km == m and (rep_filter is None or kr == rep_filter)]
        for key in keys:
            runs = groups[key]
            rep = key[1]
            ss = [r["summary"] for r in runs]
            lasts = [r["rounds"][-1] for r in runs]
            seeds = sorted({_seed_of(r) for r in runs})
            row = {
                "method": m, "representation": rep, "seeds": len(seeds),
                "seed_list": seeds,
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
                "edge_base_build_sec": _ms([s.get("edge_base_build_sec") for s in ss]),
                "edge_online_sec": _ms([s.get("edge_online_sec") for s in ss]),
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


def write_markdown(objs, out_path, dataset="cifar100", rounds=5, note=None):
    lines = [f"# {dataset} Streaming 结果汇总", "",
             f"> {rounds} 轮，Initial Base 20%，代表预算 980（最终 50:1），INT8。"
             " 数值为均值±标准差（种子 42/43/44）。", ""]
    if note:
        lines += [f"> {note}", ""]
    iid = aggregate(objs, "iid", methods=ORDER, append_extra=False)
    lines += ["## 1. IID", "",
              "| 方法 | 表示 | 种子 | 初始 | 最终 | 在线平均 | 最终 F1 | INT8 | 累计通信 | 方差比 | 对齐误差 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in iid:
        lines.append(
            f"| {r['method']} | {r['representation']} | {r['seeds']} | "
            f"{_fmt(r['initial'])} | "
            f"{_fmt(r['final'])} | {_fmt(r['online'])} | {_fmt(r['f1'])} | "
            f"{_fmt(r['int8'])} | {r['comm_kb']:.0f} KB | "
            f"{_fmt(r['var_ratio'], 3)} | {_fmt(r['align_err'], 3)} |")
    timed = [r for r in iid
             if r["edge_base_build_sec"] is not None
             or r["edge_online_sec"] is not None]
    if timed:
        lines += ["", "### 1c. 边缘耗时（秒，分开报告）", "",
                  "| 方法 | 表示 | 一次性 Base 构建 | 80 轮在线更新 | 旧口径合计 |",
                  "|---|---|---|---|---|"]
        for r in timed:
            lines.append(
                f"| {r['method']} | {r['representation']} | "
                f"{_fmt(r['edge_base_build_sec'], 1)} | "
                f"{_fmt(r['edge_online_sec'], 1)} | {_fmt(r['edge_sec'], 1)} |")
    ab = [r for r in aggregate(objs, "iid") if r["method"] not in ORDER]
    if ab:
        seeds_txt = "单种子" if ab[0]["seeds"] == 1 else f"{ab[0]['seeds']} 种子"
        lines += ["", f"## 1b. 消融（{seeds_txt}）", "",
                  "| 方法 | 表示 | 最终 | 在线平均 | INT8 | 累计通信 | 预算 |",
                  "|---|---|---|---|---|---|---|"]
        for r in ab:
            lines.append(f"| {r['method']} | {r['representation']} | "
                         f"{_fmt(r['final'])} | "
                         f"{_fmt(r['online'])} | {_fmt(r['int8'])} | "
                         f"{r['comm_kb']:.0f} KB | {r['n_reps']} |")
    ls = aggregate(objs, "label_shift")
    lines += ["", "## 2. Label Shift", "",
              "| 方法 | 表示 | 最终 | 当前类 | 历史类 | 累计通信 |", "|---|---|---|---|---|---|"]
    for r in ls:
        lines.append(f"| {r['method']} | {r['representation']} | "
                     f"{_fmt(r['final'])} | {_fmt(r.get('current'))} | "
                     f"{_fmt(r.get('historical'))} | {r['comm_kb']:.0f} KB |")
    if not ls:
        lines.append("| （无结果，待重跑） | — | — | — | — | — |")
    cs = aggregate(objs, "covariate_shift")
    lines += ["", "## 3. Covariate Shift", "",
              "| 方法 | 表示 | 原域 | 当前域 | 累计通信 |", "|---|---|---|---|---|"]
    for r in cs:
        lines.append(f"| {r['method']} | {r['representation']} | "
                     f"{_fmt(r.get('clean'))} | "
                     f"{_fmt(r.get('shifted'))} | {r['comm_kb']:.0f} KB |")
    if not cs:
        lines.append("| （无结果，待重跑） | — | — | — | — |")
    off = [o for o in objs if o["summary"].get("scenario") == "offline_non_causal"]
    if off:
        lines += ["", "## 4. 非因果离线参考（全量 49k 一次性基线）", "",
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
    ap.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"],
                    help="必须指定数据集，避免把不同数据集混进同一张表")
    ap.add_argument("--rounds", type=int, default=5,
                    help="只统计该轮数的结果（长流实验用 392）")
    ap.add_argument("--tag", default=None,
                    help="只统计方法名以该后缀结尾的结果，并在表中去掉该后缀")
    ap.add_argument("--note", default=None, help="追加到表头的说明")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    objs = load_all(os.path.join(ROOT, args.results))
    objs = [o for o in objs
            if o["summary"].get("dataset") == args.dataset
            and (o["summary"].get("scenario") == "offline_non_causal"
                 or o["summary"].get("num_rounds") == args.rounds)]
    if args.tag:
        objs = [o for o in objs
                if o["summary"]["method"].endswith(args.tag)]
        for o in objs:
            o["summary"]["method"] = o["summary"]["method"][:-len(args.tag)]
    out = args.out or f"vae_streaming/results/{args.dataset}/main_table.md"
    p = os.path.join(ROOT, out)
    write_markdown(objs, p, dataset=args.dataset, rounds=args.rounds,
                   note=args.note)
    print(open(p).read())


if __name__ == "__main__":
    main()
