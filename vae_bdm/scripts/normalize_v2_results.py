"""补齐 v2 结果文件中的标准字段别名（不改变数值）"""
import glob
import json
import os

from vae_bdm.scripts._common import ROOT


def main():
    pats = [os.path.join(ROOT, "vae_bdm", "results", "v2", "cifar*", "*.json"),
            os.path.join(ROOT, "vae_bdm", "results", "v2", "ablations", "*.json")]
    n_files = n_rows = 0
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            d = json.load(open(p))
            rows = d.get("rows")
            if not rows:
                continue
            for r in rows:
                r.setdefault("distillation_seconds", r.get("distill_sec"))
                r.setdefault("cloud_training_seconds", r.get("cloud_train_sec"))
                r.setdefault("best_distillation_step", r.get("best_step"))
                cs = r.get("cloud_seeds")
                r.setdefault("cloud_seed", int(cs[0]) if cs and len(cs) == 1 else None)
                r.setdefault("accuracy", r.get("acc_mean"))
                r.setdefault("macro_f1", r.get("f1_mean"))
                n_rows += 1
            with open(p, "w") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
            n_files += 1
    print(f"规范化 {n_files} 个文件 / {n_rows} 行")


if __name__ == "__main__":
    main()
