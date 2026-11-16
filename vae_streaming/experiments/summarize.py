"""汇总 vae_streaming/results 下的 JSON 为主表 CSV"""
import argparse
import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIELDS = ["dataset", "scenario", "method", "seed", "causal", "num_rounds",
          "n_train_used", "initial_acc", "final_acc", "final_acc_std",
          "avg_online_acc", "final_f1", "avg_online_f1",
          "final_test_int8_acc", "initial_upload_bytes",
          "cumulative_upload_bytes", "sync_mode", "quantization", "n_reps",
          "update_fraction", "patch_fraction", "stats_scope",
          "raw_replay_capacity", "replay_bytes", "edge_train_sec",
          "edge_total_sec", "eval_encode_sec_total", "int8_eval_sec_total",
          "cloud_total_sec", "peak_gpu_gb",
          "stats_seen_final", "expected_seen_final",
          "cumulative_seen_samples", "effective_stats_samples",
          "replay_samples_final",
          "final_alignment_err", "final_alignment_improvement",
          "final_class_center_improvement",
          "final_fit_class_center_err",
          "final_heldout_class_center_err", "final_group_var_ratio",
          "final_class_mean_distance", "final_local_unsynced_max",
          "vae_ckpt", "vae_accessible", "base_cache_signature",
          "protocol_source"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="vae_streaming/results")
    ap.add_argument("--out", default="vae_streaming/results/summary.csv")
    args = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, args.results, "**", "*.json"),
                              recursive=True)):
        if "_stale_pre_audit" in p.split(os.sep):
            continue
        if p.endswith("summary.json"):
            continue
        try:
            obj = json.load(open(p))
        except Exception:
            continue
        if "summary" not in obj:
            continue
        s = obj["summary"]
        row = {k: s.get(k) for k in FIELDS}
        row["causal"] = s.get("causal", True)
        row["path"] = os.path.relpath(p, ROOT)
        rows.append(row)
    if not rows:
        print("没有可汇总的结果")
        return
    df = pd.DataFrame(rows).sort_values(["dataset", "scenario", "method", "seed"])
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
