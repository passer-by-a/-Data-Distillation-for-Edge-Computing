"""校验 Pixel / VAE 逐 seed 配对，并导出配对小表。

检查项：
  - (dataset, method, representation, seed, tag) 唯一，不把文件数当 seed 数
  - 同一数据集内 Base 指纹、流指纹、round_sizes 完全一致
  - 每条轨迹完成全部轮次（completed_rounds == num_rounds 且非 partial）
  - 云端训练预算（代表数、轮数、更新比例、同步/量化、head、每轮 epoch）一致
  - 每个 (dataset, method, seed) 恰好各有 1 条指定 tag 的 Pixel 与 VAE 记录
输出：逐 seed 的末轮准确率、在线平均与配对差。
"""
import argparse
import glob
import json
import os
import statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

DEFAULT_VAE_TAG = {"cifar10": "sup80v3", "cifar100": "sup80v2"}
CORE_METHODS = ("dm", "exstream", "online_reservoir")


def load_records(results_dir, datasets):
    recs = []
    for p in glob.glob(os.path.join(results_dir, "**", "*.json"),
                       recursive=True):
        if "_stale_pre_audit" in p.split(os.sep):
            continue
        try:
            with open(p) as f:
                o = json.load(f)
        except Exception:
            continue
        s = o.get("summary")
        if not s or s.get("scenario") != "iid" or s.get("partial"):
            continue
        if s.get("dataset") not in datasets:
            continue
        if "rounds" not in o or "stream" not in o:
            continue
        o["_path"] = p
        recs.append(o)
    return recs


def representation_of(o):
    s = o["summary"]
    return str(s.get("representation")
               or (o.get("config") or {}).get("representation")
               or "vae")


def tag_of(o):
    return str((o.get("config") or {}).get("_run_tag") or "")


def method_of(o):
    m = o["summary"]["method"]
    t = tag_of(o)
    if t and m.endswith("_" + t):
        return m[:-(len(t) + 1)]
    return m


def seed_of(o):
    return int(o["summary"].get("seed", 0) or 0)


def budget_of(o):
    s = o["summary"]
    c = (o.get("config") or {}).get("cloud", {})
    return {
        "n_reps": s.get("n_reps"),
        "num_rounds": s.get("num_rounds"),
        "update_fraction": s.get("update_fraction"),
        "sync_mode": s.get("sync_mode"),
        "quantization": s.get("quantization"),
        "head": c.get("head"),
        "epochs_per_round": c.get("epochs_per_round"),
        "steps_per_round": c.get("steps_per_round"),
    }


def check_dataset(dataset, recs, errors):
    """同一数据集内所有轨迹必须共享同一数据流与训练预算。"""
    base_sha = {o["stream"].get("base_indices_sha1") for o in recs}
    # 指纹算法有代际：旧 JSON 无版本字段（v1，仅拼接索引），
    # 新 JSON 为 v2（含轮边界）。只在同一代内比较，避免跨代误报。
    by_version = defaultdict(set)
    for o in recs:
        v = int(o["stream"].get("stream_sha1_version", 1) or 1)
        by_version[v].add(o["stream"].get("stream_indices_sha1"))
    round_sizes = {tuple(o["stream"].get("round_sizes") or []) for o in recs}
    budgets = {json.dumps(budget_of(o), sort_keys=True) for o in recs}
    if len(base_sha) != 1:
        errors.append(f"{dataset}: Base 指纹不一致 {sorted(map(str, base_sha))}")
    for v, shas in sorted(by_version.items()):
        if len(shas) != 1:
            errors.append(
                f"{dataset}: 流指纹不一致（v{v}）{sorted(map(str, shas))}")
    if len(round_sizes) != 1:
        errors.append(f"{dataset}: round_sizes 不一致")
    if len(budgets) != 1:
        shown = defaultdict(list)
        for o in recs:
            shown[json.dumps(budget_of(o), sort_keys=True)].append(
                os.path.relpath(o["_path"], ROOT))
        errors.append(f"{dataset}: 云端训练预算不一致 {dict(shown)}")
    for o in recs:
        s = o["summary"]
        rounds = [r for r in (o.get("rounds") or [])
                  if int(r.get("round", -1)) >= 0]
        n_done = int(s.get("completed_rounds", len(rounds)))
        if n_done != int(s.get("num_rounds", -1)) or len(rounds) != n_done:
            errors.append(
                f"{os.path.relpath(o['_path'], ROOT)}: 未完成全部轮次 "
                f"(completed={n_done}, rounds={len(rounds)}, "
                f"num_rounds={s.get('num_rounds')})")


def fmt(ms):
    if ms is None:
        return "—"
    return f"{ms[0]:.2f}±{ms[1]:.2f}"


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    ddof = 1 if len(vals) > 1 else 0
    return statistics.mean(vals), statistics.stdev(vals) if ddof else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="vae_streaming/results")
    ap.add_argument("--datasets", default="cifar10,cifar100")
    ap.add_argument("--pixel-tag", default="pix80v3")
    ap.add_argument("--vae-tag-cifar10", default=DEFAULT_VAE_TAG["cifar10"])
    ap.add_argument("--vae-tag-cifar100", default=DEFAULT_VAE_TAG["cifar100"])
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default="vae_streaming/results/"
                                    "pixel_vs_vae_per_seed.md")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    vae_tag = {"cifar10": args.vae_tag_cifar10,
               "cifar100": args.vae_tag_cifar100}
    recs = load_records(os.path.join(ROOT, args.results), set(datasets))

    errors = []
    # 唯一性：(dataset, method, representation, seed, tag)
    seen = {}
    for o in recs:
        key = (o["summary"]["dataset"], method_of(o), representation_of(o),
               seed_of(o), tag_of(o))
        if key in seen:
            errors.append(
                "重复结果（同一 dataset/method/表示/seed/tag）："
                f"{os.path.relpath(seen[key], ROOT)} 与 "
                f"{os.path.relpath(o['_path'], ROOT)}")
        seen[key] = o["_path"]

    by_ds = defaultdict(list)
    for o in recs:
        by_ds[o["summary"]["dataset"]].append(o)
    for ds in datasets:
        if by_ds.get(ds):
            check_dataset(ds, by_ds[ds], errors)
        else:
            errors.append(f"{ds}: 未找到 iid 结果")

    lookup = {}
    for o in recs:
        lookup[(o["summary"]["dataset"], method_of(o), seed_of(o),
                representation_of(o), tag_of(o))] = o

    rows = []
    for ds in datasets:
        pixel_methods = {method_of(o) for o in recs
                         if o["summary"]["dataset"] == ds
                         and representation_of(o) == "pixel"
                         and tag_of(o) == args.pixel_tag}
        vae_methods = {method_of(o) for o in recs
                       if o["summary"]["dataset"] == ds
                       and representation_of(o) == "vae"
                       and tag_of(o) == vae_tag[ds]}
        # 三种核心方法必须成对；其余方法仅在有两侧结果时列出
        methods = sorted(set(CORE_METHODS) |
                         (pixel_methods & vae_methods))
        for m in methods:
            for seed in seeds:
                pix = lookup.get((ds, m, seed, "pixel", args.pixel_tag))
                vae = lookup.get((ds, m, seed, "vae", vae_tag[ds]))
                if pix is None or vae is None:
                    if m not in CORE_METHODS:
                        continue
                    miss = []
                    if vae is None:
                        miss.append(f"VAE[{vae_tag[ds]}]")
                    if pix is None:
                        miss.append(f"Pixel[{args.pixel_tag}]")
                    errors.append(f"{ds}/{m}/seed{seed}: 缺少 {' + '.join(miss)}")
                    continue
                # 配对两侧的数据流与预算必须一致
                for field in ("stream_indices_sha1", "base_indices_sha1",
                              "round_sizes"):
                    if pix["stream"].get(field) != vae["stream"].get(field):
                        errors.append(
                            f"{ds}/{m}/seed{seed}: {field} 两侧不一致")
                if budget_of(pix) != budget_of(vae):
                    errors.append(
                        f"{ds}/{m}/seed{seed}: 云端训练预算两侧不一致")
                ps, vs = pix["summary"], vae["summary"]
                rows.append({
                    "dataset": ds, "method": m, "seed": seed,
                    "pixel_tag": args.pixel_tag, "vae_tag": vae_tag[ds],
                    "pixel_final": ps.get("final_acc"),
                    "vae_final": vs.get("final_acc"),
                    "d_final": ps.get("final_acc") - vs.get("final_acc"),
                    "pixel_online": ps.get("avg_online_acc"),
                    "vae_online": vs.get("avg_online_acc"),
                    "d_online": ps.get("avg_online_acc") - vs.get("avg_online_acc"),
                    "pixel_int8": ps.get("final_test_int8_acc"),
                    "vae_int8": vs.get("final_test_int8_acc"),
                })

    lines = ["# Pixel vs VAE 逐 seed 配对校验", "",
             f"> 配对规则：Pixel tag = `{args.pixel_tag}`；"
             f"VAE tag = `{vae_tag['cifar10']}` (cifar10) / "
             f"`{vae_tag['cifar100']}` (cifar100)。", ""]
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] == ds]
        if not sub:
            continue
        vers = sorted({int(o["stream"].get("stream_sha1_version", 1) or 1)
                       for o in by_ds[ds]})
        ver_note = ("v2（含轮边界，新跑）" if vers == [2] else
                    "v1（旧 JSON，仅拼接索引；恢复时按兼容模式处理）"
                    if vers == [1] else f"混合 {vers}（跨代不作同值比较）")
        lines += [f"## {ds}", "", f"> 流指纹版本：{ver_note}", "",
                  "| 方法 | seed | VAE tag | Pixel tag | VAE 末轮 | Pixel 末轮 | "
                  "Δ末轮 (P−V) | VAE 在线 | Pixel 在线 | Δ在线 (P−V) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in sub:
            lines.append(
                f"| {r['method']} | {r['seed']} | {r['vae_tag']} | "
                f"{r['pixel_tag']} | {r['vae_final']:.2f} | "
                f"{r['pixel_final']:.2f} | {r['d_final']:+.2f} | "
                f"{r['vae_online']:.2f} | {r['pixel_online']:.2f} | "
                f"{r['d_online']:+.2f} |")
        lines += ["", "按方法汇总配对差（均值±标准差，P−V）：", "",
                  "| 方法 | n seeds | Δ末轮 | Δ在线 |", "|---|---|---|---|"]
        for m in sorted({r["method"] for r in sub}):
            ms = [r for r in sub if r["method"] == m]
            lines.append(
                f"| {m} | {len(ms)} | "
                f"{fmt(mean_std([r['d_final'] for r in ms]))} | "
                f"{fmt(mean_std([r['d_online'] for r in ms]))} |")
        lines.append("")

    core = [r for r in rows if r["method"] in CORE_METHODS]
    if core:
        per_run = [abs(r["vae_int8"] - r["vae_final"]) for r in core]
        per_run += [abs(r["pixel_int8"] - r["pixel_final"]) for r in core]
        groups = defaultdict(list)
        for r in core:
            groups[(r["dataset"], r["method"], "vae")].append(
                r["vae_final"] - r["vae_int8"])
            groups[(r["dataset"], r["method"], "pixel")].append(
                r["pixel_final"] - r["pixel_int8"])
        seed_means = [abs(statistics.mean(v)) for v in groups.values()]
        lines += ["## INT8 测试输入切换差（核心方法）", "",
                  "同一云端模型（INT8 回传训练）下，只切换测试输入精度：",
                  f"按 seed 平均的末轮准确率差最大 **{max(seed_means):.2f} pp**，"
                  f"逐 seed 最大 {max(per_run):.2f} pp"
                  f"（n={len(per_run)} 条轨迹，方法={', '.join(CORE_METHODS)}）。",
                  ""]

    lines += ["## 校验结果", ""]
    if errors:
        lines += [f"- FAIL: {e}" for e in errors]
    else:
        lines += ["- PASS: 唯一性、Base/流指纹、round_sizes、完成轮数、"
                  "云端预算与 42/43/44 配对全部通过"]

    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
