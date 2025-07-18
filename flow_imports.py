# common_imports.py

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
# import seaborn as sns
from tqdm import tqdm
from tqdm.auto import tqdm as pbar  # pbar is just an alias for convenience
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn

import os
import sys
import random
import time

# Optional: configure default styles
# plt.style.use('seaborn-v0_8')
# sns.set_theme()

__all__ = [
    "np",
    "pd",
    "torch",
    "plt",
    "animation",
    "FuncAnimation",
    "tqdm",
    "pbar",
    "TensorDataset",
    "DataLoader",
    "nn",
    "os",
    "sys",
    "random",
    "time"
]
