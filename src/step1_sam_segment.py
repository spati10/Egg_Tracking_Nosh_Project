"""TASK 1 - Pretrained segmentation with SAM (produces the 'ground truth').

Usage:
  python src/step1_sam_segment.py --images data/raw --out outputs/sam \
         --ckpt checkpoints/sam_vit_b_01ec64.pth --model vit_b
"""
import argparse
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import list_images, stem, overlay


def ellipse_fit_ratio(mask):
    """How egg-like is the mask? 1.0 = perfect ellipse."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return 0.0
    try:
        (_, _), (MA, ma), _ = cv2.fitEllipse(c)
        ea = np.pi * MA * ma / 4
        a = cv2.contourArea(c)
        return min(a, ea) / max(a, ea, 1.0)
    except cv2.error:
        return 0.0



def iou_bool(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def pick_eggs(masks, shape, min_frac, max_frac, min_ellipse, max_keep=1):
    H, W = shape
    cands = []
    for m in masks:
        seg = m["segmentation"]
        frac = seg.sum() / (H * W)
        if not (min_frac <= frac <= max_frac):
            continue
        er = ellipse_fit_ratio(seg)
        if er < min_ellipse:
            continue
        # Weight predicted IoU, stability, and ellipse fit
        score = m["predicted_iou"] * m["stability_score"] * max(er, 0.1)
        cands.append((score, seg))
    cands.sort(key=lambda x: -x[0])
    keep = []
    for score, seg in cands:  # NMS: drop masks that overlap a better one
        if all(iou_bool(seg, k) < 0.3 for k in keep):
            keep.append(seg)
            if max_keep > 0 and len(keep) >= max_keep:
                break
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--out", default="outputs/sam")
    ap.add_argument("--ckpt", default="checkpoints/sam_vit_b_01ec64.pth")
    ap.add_argument("--model", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--min_frac", type=float, default=0.01)
    ap.add_argument("--max_frac", type=float, default=0.80)
    ap.add_argument("--min_ellipse", type=float, default=0.55)
    ap.add_argument("--max_keep", type=int, default=1, help="Max eggs per image (1 for single egg dataset)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (CUDA not detected)")

    sam = sam_model_registry[args.model](checkpoint=args.ckpt).to(device)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.86,
                                    stability_score_thresh=0.90, min_mask_region_area=200)

    flagged = []
    for path in tqdm(list_images(args.images)):
        img = cv2.imread(path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = gen.generate(rgb)
        eggs = pick_eggs(masks, img.shape[:2], args.min_frac, args.max_frac, args.min_ellipse, args.max_keep)
        if not eggs:  # fallback: relax ellipse and expand max_frac for close-ups
            eggs = pick_eggs(masks, img.shape[:2], args.min_frac, 0.90, 0.0, args.max_keep)
            flagged.append(stem(path))
        union = np.zeros(img.shape[:2], np.uint8)
        for e in eggs:
            union[e] = 255
        cv2.imwrite(os.path.join(args.out, "masks", stem(path) + ".png"), union)
        cv2.imwrite(os.path.join(args.out, "overlays", stem(path) + ".jpg"), overlay(img, union > 0))

    with open(os.path.join(args.out, "flagged_check_manually.txt"), "w") as f:
        f.write("\n".join(flagged))
    print(f"Done. {len(flagged)} images used the fallback -> check them manually.")


if __name__ == "__main__":
    main()
