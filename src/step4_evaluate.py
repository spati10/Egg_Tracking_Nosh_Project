"""TASK 3 - Quantitative + qualitative evaluation on the EVAL split.
Requires: outputs/pred_final (made by the --final run in step 3).

Usage: python src/step4_evaluate.py
Creates outputs/eval/: per_image_metrics.csv, summary.txt, iou_hist.png,
iou_vs_brightness.png, iou_vs_size.png, worst_cases.png, error_breakdown.png
"""
import argparse
import json
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import list_images, stem, read_mask, metrics


def error_map(img, gt, pred):
    out = img.copy()
    out[gt & pred] = (0.5 * out[gt & pred] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)   # TP green
    out[~gt & pred] = (0.4 * out[~gt & pred] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)  # FP red
    out[gt & ~pred] = (0.4 * out[gt & ~pred] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)  # FN blue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--gt", default="outputs/sam/masks")
    ap.add_argument("--pred", default="outputs/pred_final")
    ap.add_argument("--split_file", default="outputs/split.json")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    paths = {stem(p): p for p in list_images(args.images)}
    rows = []
    for nm in json.load(open(args.split_file))["eval"]:
        gt = read_mask(os.path.join(args.gt, nm + ".png"))
        pr = read_mask(os.path.join(args.pred, nm + ".png"))
        if gt is None or pr is None:
            continue
        img = cv2.imread(paths[nm])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = metrics(gt, pr)
        m.update(name=nm, brightness=float(hsv[..., 2].mean()), egg_frac=float(gt.mean()),
                 fp_px=int((~gt & pr).sum()), fn_px=int((gt & ~pr).sum()),
                 empty_pred=bool(pr.sum() == 0))
        rows.append(m)
    df = pd.DataFrame(rows).sort_values("iou")
    df.to_csv(os.path.join(args.out, "per_image_metrics.csv"), index=False)

    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(df[["iou", "dice", "precision", "recall", "pixel_acc"]].describe().round(4).to_string())
        f.write(f"\n\nIoU>=0.9: {(df.iou >= 0.9).mean():.2%}  IoU<0.5: {(df.iou < 0.5).mean():.2%}")
        f.write(f"\nEmpty predictions: {df.empty_pred.sum()}\n")
    print(open(os.path.join(args.out, "summary.txt")).read())

    plt.figure(figsize=(6, 4)); plt.hist(df.iou, bins=20, color="steelblue", edgecolor="k")
    plt.xlabel("IoU"); plt.ylabel("# images"); plt.title("IoU distribution (eval)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "iou_hist.png"), dpi=150); plt.close()

    for col, fn in [("brightness", "iou_vs_brightness.png"), ("egg_frac", "iou_vs_size.png")]:
        plt.figure(figsize=(6, 4)); plt.scatter(df[col], df.iou, s=14)
        plt.xlabel(col); plt.ylabel("IoU"); plt.title(f"IoU vs {col}")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, fn), dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(df.recall, df.precision, c=df.iou, cmap="viridis", s=18)
    plt.colorbar(label="IoU"); plt.xlabel("Recall (low = under-seg)"); plt.ylabel("Precision (low = over-seg)")
    plt.title("Error type per image"); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "error_breakdown.png"), dpi=150); plt.close()

    worst = df.head(args.k)
    fig, ax = plt.subplots(len(worst), 4, figsize=(12, 3 * len(worst)))
    ax = np.atleast_2d(ax)
    for r, (_, row) in enumerate(worst.iterrows()):
        nm = row["name"]
        img = cv2.imread(paths[nm])
        gt = read_mask(os.path.join(args.gt, nm + ".png"))
        pr = read_mask(os.path.join(args.pred, nm + ".png"))
        for c, (im, t) in enumerate([(img, f"{nm[:22]}"), (gt, "SAM (GT)"), (pr, "Classical"),
                                     (error_map(img, gt, pr), f"IoU={row.iou:.2f} (red=FP, blue=FN)")]):
            ax[r, c].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im, cmap="gray")
            ax[r, c].set_title(t, fontsize=8); ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "worst_cases.png"), dpi=130); plt.close()
    print("Saved plots to", args.out)


if __name__ == "__main__":
    main()
