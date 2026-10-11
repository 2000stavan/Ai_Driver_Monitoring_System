"""Frame-level and event-level metrics (INITIALPLAN §12.3)."""

from __future__ import annotations

import numpy as np

Segment = tuple[int, int]  # (start_frame, end_frame), end inclusive


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p + r and not np.isnan(p) and not np.isnan(r) else float("nan")
    return {"precision": p, "recall": r, "f1": f1}


def frame_prf(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    if valid is not None:
        pred, gt = pred[valid], gt[valid]
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    return {**prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}


def _overlaps(a: Segment, b: Segment, tol: int) -> bool:
    return a[0] - tol <= b[1] and b[0] - tol <= a[1]


def event_prf(pred: list[Segment], gt: list[Segment], tol: int = 0) -> dict[str, float]:
    """An event counts as detected if any prediction overlaps it (± tol frames), and a
    prediction counts as correct if it overlaps any ground-truth event."""
    gt_hit = sum(any(_overlaps(g, p, tol) for p in pred) for g in gt)
    pred_hit = sum(any(_overlaps(p, g, tol) for g in gt) for p in pred)
    recall = gt_hit / len(gt) if gt else float("nan")
    precision = pred_hit / len(pred) if pred else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if gt and pred and precision + recall
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1, "n_gt": len(gt), "n_pred": len(pred)}


def roc_auc(score: np.ndarray, gt: np.ndarray) -> float:
    """Threshold-free ranking quality (Mann-Whitney U). 0.5 = random, 1.0 = perfect."""
    score, gt = np.asarray(score, dtype=float), np.asarray(gt, dtype=bool)
    ok = ~np.isnan(score)
    score, gt = score[ok], gt[ok]
    n_pos, n_neg = int(gt.sum()), int((~gt).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = score.argsort(kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[gt].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
