"""Road segmentation on Indian roads: Western-only (Cityscapes) vs. IDD fine-tuned SegFormer.

    python scripts/train_road_seg.py                       # IDD Lite, SegFormer-B0, Mac (MPS)
    python scripts/train_road_seg.py --epochs 1 --limit 32 # smoke test
    python scripts/train_road_seg.py --zero-shot-only

1. Zero-shot: the Cityscapes model as-is, 19 classes merged into IDD's 7 level-1 classes.
2. Fine-tune: same backbone, new 7-class head, trained on IDD Lite train.
Both are scored on IDD Lite val (mIoU, per-class IoU, drivable-area IoU).

Writes models/road/segformer_idd_lite.pt and reports/results/road_seg_idd_lite.md.
Full IDD + SegFormer-B2 at full resolution is the Kaggle-GPU version of this script (INITIALPLAN §6.2).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd
import torch

from dms.road.segmentation import (
    IGNORE, L1_CLASSES, ConfusionMatrix, IDDLite, cityscapes_to_l1, load_segformer, predict_logits,
)
from dms.utils.config import PROJECT_ROOT

DATA = PROJECT_ROOT / "data/raw/road/idd_lite/idd20k_lite"
CACHE = PROJECT_ROOT / "models/segformer"
OUT_MODEL = PROJECT_ROOT / "models/road/segformer_idd_lite.pt"
OUT_REPORT = PROJECT_ROOT / "reports/results/road_seg_idd_lite.md"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, dev, zero_shot: bool) -> ConfusionMatrix:
    model.eval()
    cm = ConfusionMatrix(len(L1_CLASSES))
    for x, y in loader:
        pred = predict_logits(model, x.to(dev)).argmax(1)
        if zero_shot:
            pred = cityscapes_to_l1(pred)
            pred[pred == IGNORE] = 0  # every Cityscapes class is mapped; kept for safety
        cm.update(pred, y.to(dev))
    return cm


def subset(ds, n):
    return torch.utils.data.Subset(ds, range(min(n, len(ds)))) if n else ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--size", type=int, nargs=2, default=[288, 416], metavar=("H", "W"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=0, help="use only N images per split (smoke test)")
    ap.add_argument("--zero-shot-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random

    random.seed(args.seed)
    dev = device()
    size = tuple(args.size)
    train_ds = subset(IDDLite(args.data, "train", size, train=True), args.limit)
    val_ds = subset(IDDLite(args.data, "val", size), args.limit)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch)
    print(f"device {dev}, train {len(train_ds)}, val {len(val_ds)}, input {size}")

    results = {}
    zs = load_segformer(args.checkpoint, cache_dir=CACHE).to(dev)
    results["Cityscapes only (zero-shot, Western)"] = evaluate(zs, val_dl, dev, zero_shot=True)
    print("zero-shot mIoU", np.nanmean(results["Cityscapes only (zero-shot, Western)"].iou()).round(3))
    del zs

    log = []
    if not args.zero_shot_only:
        model = load_segformer(args.checkpoint, num_labels=len(L1_CLASSES), cache_dir=CACHE).to(dev)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                               drop_last=True)  # workers are slower than in-process on macOS
        head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
        body = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
        opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}],
                                weight_decay=0.01)
        total = args.epochs * len(train_dl)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (1 - s / total) ** 0.9)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
        best, t0 = -1.0, time.time()
        for ep in range(args.epochs):
            model.train()
            run = 0.0
            for x, y in train_dl:
                x, y = x.to(dev), y.to(dev)
                loss = loss_fn(predict_logits(model, x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                run += loss.item()
            miou = float(np.nanmean(evaluate(model, val_dl, dev, zero_shot=False).iou()))
            log.append({"epoch": ep + 1, "train_loss": run / len(train_dl), "val_miou": miou})
            print(f"epoch {ep + 1:3d}  loss {run / len(train_dl):.3f}  val mIoU {miou:.3f}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            if miou > best:
                best = miou
                OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "checkpoint": args.checkpoint, "size": size,
                            "classes": L1_CLASSES, "epoch": ep + 1, "val_miou": miou}, OUT_MODEL)
        model.load_state_dict(torch.load(OUT_MODEL, map_location=dev)["state_dict"])
        results["IDD Lite fine-tuned (Indian)"] = evaluate(model, val_dl, dev, zero_shot=False)

    table = pd.DataFrame({name: cm.iou() for name, cm in results.items()}, index=L1_CLASSES).T
    table.insert(0, "mIoU", table.mean(axis=1))
    table.insert(1, "pixel_acc", [cm.pixel_acc() for cm in results.values()])
    lines = ["# Road segmentation on Indian roads - IDD Lite val", "",
             f"Model: `{args.checkpoint}`, input {size[1]}x{size[0]}, {len(train_ds)} train / {len(val_ds)} val images, "
             f"device {dev}.", "",
             "IoU per IDD level-1 class (zero-shot = Cityscapes classes merged into IDD level-1):", "",
             table.round(3).to_markdown(), ""]
    if log:
        best_ep = max(log, key=lambda r: r["val_miou"])
        lines += [f"Fine-tuning: {args.epochs} epochs, best val mIoU {best_ep['val_miou']:.3f} at epoch {best_ep['epoch']}. "
                  "Note: val is used for checkpoint selection (IDD Lite has no public test labels), so the "
                  "fine-tuned number is slightly optimistic.", "",
                  pd.DataFrame(log).round(4).to_markdown(index=False)]
    if not args.limit:
        OUT_REPORT.write_text("\n".join(lines))
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
