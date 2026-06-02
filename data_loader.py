"""
data_loader.py — DRB Stream Temperature Dataset Loader

Provides a PyTorch Dataset and DataLoader utilities for the Delaware River Basin
stream temperature forecasting task.

Data files (.npz) are NOT included in this repository. Download them from the
Google Drive link in README.md and place them in the data/ folder.

Quick start (single site):
    X, Y, _ = load_site_npz(1573, data_dir='data/', split='finetune')
    # Normalize X yourself before passing to the dataset
    train_loader, val_loader = get_dataloaders(X, Y, sequence_length=360,
                                               forecast_horizon=1)
 
Quick start (all sites):
    Xs, Ys = [], []
    for sid in SITE_IDS:
        X, Y, _ = load_site_npz(sid, data_dir='data/', split='finetune')
        Xs.append(X)
        Ys.append(Y)
    # Normalize Xs yourself before passing to the dataset
    train_loader, val_loader = get_dataloaders(Xs, Ys, sequence_length=360,
                                               forecast_horizon=1)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Feature metadata (matches config.yaml)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    'tmin',                  # daily min air temperature (°C, GridMET)
    'tmax',                  # daily max air temperature (°C, GridMET)
    'srad',                  # solar radiation (W/m², GridMET)
    'pr',                    # precipitation (mm, GridMET)
    'vs',                    # wind speed (m/s, GridMET)
    'rmax',                  # relative humidity max (%, GridMET)
    'rmin',                  # relative humidity min (%, GridMET)
    'rmean',                 # relative humidity mean (%, GridMET)
    'spillway',              # spillway release from the site's closest reservoir (cms)
    'releases',              # controlled release from the site's closest reservoir (cms)
    'seg_slope',             # stream segment slope (static site attribute)
    'seg_elev',              # segment elevation (static site attribute)
    'seg_width',             # segment width (static site attribute)
    'seg_length',            # segment length (static site attribute)
    'stream_temp',           # previous day's stream temp — ar1 feature, MASKED at forecast steps
]

AR1_COL = 14  # index of stream_temp in X; this column is zeroed at forecast steps

SITE_IDS = [1573, 1571, 1565, 1450, 1641]

def load_site_npz(site_id, data_dir='data/', split='finetune'):
    """Load one site's .npz file and return (X, Y, mask).

    Parameters
    ----------
    site_id : int
        USGS site ID (one of 1573, 1571, 1565, 1450, 1641).
    data_dir : str
        Directory containing the .npz files.
    split : {'pretrain', 'finetune'}
        Which time period to load.
        - 'pretrain'  : 1985-05-01 to 2020-04-01  (~12,754 daily rows)
        - 'finetune'  : 1985-05-01 to 2021-04-14  (~13,132 daily rows,
                        includes dwallin-infilled targets)

    Returns
    -------
    X : np.ndarray, shape (T, 15)
        Input features (not normalized — normalize before passing to the dataset).
    Y : np.ndarray, shape (T,)
        Target: next-day stream temperature (°C).
    mask : np.ndarray, shape (T,) or None
        Boolean array where True = target is dwallin-infilled (not a real
        observation). Only returned for split='finetune'; None for 'pretrain'.
    """
    path = os.path.join(data_dir, f'{site_id}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Download the .npz files from Google Drive (see README.md) "
            f"and place them in '{data_dir}'."
        )
    data = np.load(path, allow_pickle=True)

    if split == 'pretrain':
        return data['pretrain_X'].astype(np.float32), \
               data['pretrain_Y'].astype(np.float32), \
               None
    elif split == 'finetune':
        return data['finetune_X'].astype(np.float32), \
               data['finetune_Y'].astype(np.float32), \
               data['finetune_Y_mask'].astype(bool)
    else:
        raise ValueError(f"split must be 'pretrain' or 'finetune', got '{split}'")


class StreamTempDataset(Dataset):
    """Sliding-window dataset for one or more stream temperature sites.

    Each item is a (seq, target) tuple:
      - seq    : (sequence_length + forecast_horizon, num_features)
                 The full window of input features. The stream_temp column
                 (AR1_COL) is zeroed in the last forecast_horizon steps to
                 prevent data leakage — the model cannot see the true future
                 temperature it is trying to predict.
      - target : (forecast_horizon,) — true stream temperatures to predict

    Multi-site usage: pass X and Y as lists (one array per site). Windows are
    drawn independently per site and then pooled — time boundaries between
    sites are never crossed.

    Normalization: X is expected to be normalized before being passed here.
    """

    def __init__(self, X, Y, sequence_length=360, forecast_horizon=1,
                 split='train', val_ratio=0.2, ar1_col=AR1_COL):
        """
        Parameters
        ----------
        X : np.ndarray (T, F) or list of np.ndarray
            Input features. Normalize before passing.
        Y : np.ndarray (T,) or list of np.ndarray
            Targets.
        sequence_length : int
            Number of historical days fed as context (default 360).
        forecast_horizon : int
            Number of days ahead to predict, 1–8 (default 1).
        split : {'train', 'val'}
            Which chronological portion to expose.
        val_ratio : float
            Fraction of windows reserved for validation (default 0.2).
        ar1_col : int
            Column index of stream_temp to zero out at forecast steps.
        """
        assert 1 <= forecast_horizon <= 8, \
            f"forecast_horizon must be 1–8, got {forecast_horizon}"
        assert split in ('train', 'val'), \
            f"split must be 'train' or 'val', got '{split}'"

        if isinstance(X, np.ndarray):
            X, Y = [X], [Y]

        self.seq_len = sequence_length
        self.horizon = forecast_horizon
        self.ar1_col = ar1_col

        # Build flat index: list of (site_X, site_Y, window_start)
        self._samples = []
        for x, y in zip(X, Y):
            assert x.shape[0] == y.shape[0], "X and Y must have the same length"
            n_windows = len(y) - sequence_length - forecast_horizon + 1
            if n_windows <= 0:
                continue
            val_start = int(n_windows * (1 - val_ratio))
            indices = range(0, val_start) if split == 'train' else range(val_start, n_windows)
            for i in indices:
                self._samples.append((x, y, i))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        x, y, i = self._samples[idx]
        window = x[i : i + self.seq_len + self.horizon].copy()
        window[-self.horizon:, self.ar1_col] = 0.0  # prevent AR1 leakage at forecast steps
        seq = torch.from_numpy(window)
        target = torch.from_numpy(y[i + self.seq_len : i + self.seq_len + self.horizon])
        return seq, target

def get_dataloaders(X, Y, sequence_length=360, forecast_horizon=1,
                    batch_size=32, val_ratio=0.2, num_workers=0):
    """Create train and validation DataLoaders.

    Parameters
    ----------
    X : np.ndarray or list of np.ndarray
        Input features (normalize before calling this).
    Y : np.ndarray or list of np.ndarray
        Targets.
    sequence_length : int
    forecast_horizon : int, 1–8
    batch_size : int
    val_ratio : float
    num_workers : int

    Returns
    -------
    train_loader, val_loader : DataLoader, DataLoader
    """
    kwargs = dict(sequence_length=sequence_length, forecast_horizon=forecast_horizon,
                  val_ratio=val_ratio)
    train_ds = StreamTempDataset(X, Y, split='train', **kwargs)
    val_ds   = StreamTempDataset(X, Y, split='val',   **kwargs)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                         pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    return train_loader, val_loader
