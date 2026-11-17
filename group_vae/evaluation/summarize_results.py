"""汇总 Group-VAE 与基线结果"""
import argparse
import glob
import json
import os

from group_vae.config_utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="group_vae/configs/cifar10.yaml")
    ap.add_argument("--ratio", type=float, default=10.0)
    a = ap.parse_args()
    cfg = load_config(a.config)
    R = cfg["results_dir"]
    rows = []
    for p in sorted(glob.glob(os.path.join(R, "*_cloud.json"))):
        d = json.load(open(p))
        if abs(float(d["meta"].get("ratio", a.ratio)) - a.ratio) > 1e-6:
            continue
        m = d["modes"].get("fp32")
        if not m:
            continue
        rows.append({"method": d["tag"], "fusion": d["meta"].get("fusion"),
                     "dm": d["meta"].get("lambda_dm"),
                     "acc": m["acc_mean"], "std": m["acc_std"],
                     "f1": m["f1_mean"],
                     "bytes_int8": d["modes"].get("int8", {}).get(
                         "payload", {}).get("total")})
    bp = os.path.join(R, "baselines", f"baselines_r{a.ratio:g}.json")
    if os.path.exists(bp):
        for r in json.load(open(bp))["rows"]:
            rows.append({"method": r["method"], "fusion": "latent",
                         "dm": None, "acc": r["acc_mean"],
                         "std": r["acc_std"], "f1": r["f1_mean"],
                         "bytes_int8": r.get("upload_bytes")})
    rows.sort(key=lambda x: -x["acc"])
    print(f"\n## {cfg['dataset']} @ {a.ratio:g}:1\n")
    print("| method | fusion | ACC | F1 | INT8 bytes |")
    print("|---|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['method']} | {r['fusion']} | {r['acc']:.2f}±{r['std']:.2f} | "
              f"{r['f1']:.2f} | {r['bytes_int8']} |")
    out = os.path.join(R, f"summary_r{a.ratio:g}.csv")
    with open(out, "w") as f:
        f.write("method,fusion,acc,std,f1,bytes_int8\n")
        for r in rows:
            f.write(f"{r['method']},{r['fusion']},{r['acc']},{r['std']},"
                    f"{r['f1']},{r['bytes_int8']}\n")
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
