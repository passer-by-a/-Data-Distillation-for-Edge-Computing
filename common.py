"""共享工具: 路径、随机种子、指标"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1(pred, y, num_classes):
    from sklearn.metrics import f1_score
    return f1_score(y.cpu().numpy(), pred.cpu().numpy(),
                    average="macro", labels=list(range(num_classes)),
                    zero_division=0) * 100
