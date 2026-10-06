"""Bootstrap statistics.

All comparisons between prompt conditions are *paired*: the same bootstrap
resamples of images are used for every condition, so the confidence
interval of a difference reflects only the effect of the prompts.
"""
import numpy as np
from scipy.stats import rankdata, spearmanr


def bootstrap_indices(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def auroc_rows(y, s):
    """Rank-based AUROC (Mann-Whitney U), vectorised over rows; ties handled.

    y, s: arrays of shape (n,) or (B, n). Rows with a single class give NaN.
    """
    y = np.atleast_2d(y).astype(bool)
    s = np.atleast_2d(s).astype(np.float64)
    ranks = rankdata(s, axis=1)
    n_pos = y.sum(1)
    n_neg = y.shape[1] - n_pos
    sum_pos = (ranks * y).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    auc[(n_pos == 0) | (n_neg == 0)] = np.nan
    return auc


def auroc(y, s):
    return float(auroc_rows(y, s)[0])


def boot_auroc(y, s, idx):
    """Point estimate plus the bootstrap distribution (length B)."""
    return auroc(y, s), auroc_rows(y[idx], s[idx])


def ci(dist, alpha=0.05):
    d = np.asarray(dist, dtype=np.float64)
    d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def paired_diff(point_a, dist_a, point_b, dist_b):
    """Difference A - B with percentile CI and a two-sided bootstrap p-value."""
    diff = np.asarray(dist_a) - np.asarray(dist_b)
    diff = diff[~np.isnan(diff)]
    lo, hi = ci(diff)
    if diff.size == 0:
        p = float("nan")
    else:
        p = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
    return point_a - point_b, lo, hi, p


def boot_spearman(x, y, idx):
    rho, p = spearmanr(x, y)
    dist = np.empty(idx.shape[0])
    for b in range(idx.shape[0]):
        xb, yb = x[idx[b]], y[idx[b]]
        dist[b] = spearmanr(xb, yb)[0] if np.std(xb) > 0 and np.std(yb) > 0 else np.nan
    return float(rho), float(p), dist


def fmt_ci(point, lo, hi, nd=3):
    if point is None or np.isnan(point):
        return "n/a"
    return f"{point:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
