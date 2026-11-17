"""IID 数据流实验入口

python -m vae_streaming.experiments.run_iid \
    --config vae_streaming/configs/cifar100.yaml \
    --methods online_kmeans,online_moment,dm --seeds 42
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vae_streaming.experiments.cli import run_cli  # noqa: E402


if __name__ == "__main__":
    run_cli("iid", ["online_kmeans", "online_moment", "dm"])
