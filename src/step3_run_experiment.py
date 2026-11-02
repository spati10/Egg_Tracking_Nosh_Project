"""TASK 2c - Run ONE experiment and append it to the experiment log.

Dev experiments (repeat many times, change --params each time):
  python src/step3_run_experiment.py --name E01_baseline_hsv \
      --desc "HSV threshold: low saturation + high value" \
      --params '{"method":"hsv","s_max":60,"v_min":140}'

Final run on the held-out eval split (ONCE, at the very end):
  python src/step3_run_experiment.py --name FINAL --desc "final algo" \
      --params '{...best params...}' --split eval --final
"""
import argparse
import csv
import json
import os
import sys
import time
import cv2
import numpy as np
from tqdm import tqdm
from utils import list_images, stem, read_mask, metrics
from egg_seg import segment, DEFAULT


def parse_params(p_str):
    if not p_str or p_str.strip() in ("{}", ""):
        return {}
    import ast
    try:
        return json.loads(p_str)
    except Exception:
        pass
    try:
        return ast.literal_eval(p_str)
    except Exception:
        pass
    try:
        cleaned = p_str.replace("'", '"')
        return json.loads(cleaned)
    except Exception:
        pass

    # Fallback for Windows PowerShell stripping double quotes: {method:hsv,s_max:60,v_min:140}
    try:
        s = p_str.strip().strip("{}")
        pairs = [item.strip() for item in s.split(",") if item.strip()]
        d = {}
        for pair in pairs:
            if ":" in pair:
                k, v = pair.split(":", 1)
            elif "=" in pair:
                k, v = pair.split("=", 1)
            else:
                continue
            k = k.strip().strip("'\"")
            v = v.strip().strip("'\"")
            low = v.lower()
            if low == "true":
                v = True
            elif low == "false":
                v = False
            elif low == "none":
                v = None
            else:
                try:
                    v = float(v) if "." in v else int(v)
                except ValueError:
                    pass
            d[k] = v
        return d
    except Exception as e:
        raise ValueError(f"Could not parse --params '{p_str}': {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--split", default="dev", choices=["dev", "eval"])
    ap.add_argument("--final", action="store_true", help="required to touch the eval split")
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--gt", default="outputs/sam/masks")
    ap.add_argument("--split_file", default="outputs/split.json")
    ap.add_argument("--log", default="experiments/experiment_log.csv")
    ap.add_argument("--pred_out", default=None)
    args = ap.parse_args()

    if args.split == "eval" and not args.final:
        sys.exit("Eval split is locked. Use --final only for the final run.")

    parsed_p = parse_params(args.params)
    params = {**DEFAULT, **parsed_p}
    names = json.load(open(args.split_file))[args.split]
    paths = {stem(p): p for p in list_images(args.images)}
    pred_out = args.pred_out or ("outputs/pred_final" if args.split == "eval" else None)
    if pred_out:
        os.makedirs(pred_out, exist_ok=True)

    rows = []
    skipped_empty_gt = 0
    t0 = time.time()
    for nm in tqdm(names, desc=f"Running {args.name}"):
        gt_path = os.path.join(args.gt, nm + ".png")
        if not os.path.exists(gt_path) or nm not in paths:
            continue
        gt = read_mask(gt_path)
        if gt is None:
            continue
        if gt.sum() == 0:
            skipped_empty_gt += 1
            continue

        img = cv2.imread(paths[nm])
        if img is None:
            continue
        pred = segment(img, params)
        if pred_out:
            cv2.imwrite(os.path.join(pred_out, nm + ".png"), pred)
        rows.append(metrics(gt, pred > 0))

    if not rows:
        sys.exit(f"Error: No valid images evaluated for {args.name} (check paths/GT masks).")

    t = (time.time() - t0) / max(len(rows), 1)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    med_iou = float(np.median([r["iou"] for r in rows]))
    print(f"\n[{args.name}] split={args.split} evaluated={len(rows)} (skipped {skipped_empty_gt} empty GTs)")
    print(" ".join(f"{k}={v:.4f}" for k, v in mean.items()) + f" median_iou={med_iou:.4f} sec/img={t:.4f}")

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    new = not os.path.exists(args.log)
    with open(args.log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "name", "description", "split", "n", "mean_iou", "median_iou",
                        "mean_dice", "mean_precision", "mean_recall", "sec_per_img", "params",
                        "decision_and_reason"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), args.name, args.desc, args.split, len(rows),
                    round(mean["iou"], 4), round(med_iou, 4), round(mean["dice"], 4),
                    round(mean["precision"], 4), round(mean["recall"], 4), round(t, 4),
                    json.dumps(parsed_p), ""])


if __name__ == "__main__":
    main()
