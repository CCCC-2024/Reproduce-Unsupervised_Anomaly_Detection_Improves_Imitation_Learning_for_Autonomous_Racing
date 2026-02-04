# src/datasets/torch_dataset.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NpzMixDataset(Dataset):
    """
    Loads mix dataset from npz:
      - X: (N,H,W,3) uint8
      - y_dirty: (N,) uint8   (0=clean, 1=dirty)
    Returns:
      x: (3,img_size,img_size) float32 in [0,1]
      y: int
      idx: original index in npz
    """
    def __init__(
        self,
        npz_path: str,
        img_size: int = 64,
        only_clean: bool = False,
        limit: int | None = None,
    ):
        d = np.load(npz_path, allow_pickle=True)
        self.X = d["X"]
        self.y = d["y_dirty"]

        idx = np.arange(len(self.y))
        if only_clean:
            idx = idx[self.y == 0]
        if limit is not None:
            idx = idx[: int(limit)]
        self.idx = idx

        self.img_size = int(img_size)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k: int):
        i = int(self.idx[k])
        x = self.X[i]  # HWC uint8
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0  # CHW float

        # resize to 64x64 to match CAE Table I (4096=64*8*8)
        if x.shape[1] != self.img_size or x.shape[2] != self.img_size:
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        y = int(self.y[i])
        return x, y, i
