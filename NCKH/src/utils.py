import numpy as np
import pandas as pd

EPS = 1e-9

def safe_div(a, b):
    return a / (np.abs(b) + EPS)

def safe_log1p(x: pd.Series) -> pd.Series:
    x = x.fillna(0)
    x = np.clip(x, 0, None)
    return np.log1p(x)

def winsorize_series(s: pd.Series, q_low=0.01, q_high=0.99) -> pd.Series:
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    return s.clip(lo, hi)

def ks_statistic(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    order = np.argsort(y_prob)
    y = y_true[order]

    pos = (y == 1).astype(int)
    neg = (y == 0).astype(int)

    cum_pos = np.cumsum(pos) / max(pos.sum(), 1)
    cum_neg = np.cumsum(neg) / max(neg.sum(), 1)

    return float(np.max(np.abs(cum_pos - cum_neg)))
