"""第二阶段: Memory-aware Streaming Latent Dataset Distillation

严格因果数据流下持续更新 EdgeVAE，并用双层记忆 + 跨版本特征对齐
维护可增量同步的 1024 维代表集。第一阶段 vae_bdm/ 与 group_vae/ 不受影响。
"""
import os

# cuBLAS 可复现性（必须早于首次 CUDA 矩阵乘）
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__all__ = []

