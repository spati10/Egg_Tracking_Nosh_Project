"""TASK 2a - Leakage-aware 4:1 dev/eval split.

Leakage sources handled:
 1. Roboflow augmentations: 'img_jpg.rf.<hash>.jpg' -> same base name = same group.
 2. Near-duplicate photos (burst shots, flips) -> perceptual dHash, incl. flipped versions.
Whole GROUPS go to either dev or eval, never both.

Usage: python src/step2_split.py --images data/raw --out outputs/split.json
"""
import argparse
import json
import os
import random
import cv2
import numpy as np
from utils import list_images, stem


def dhash(gray, size=8):
    r = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (r[:, 1:] > r[:, :-1]).flatten()


def hashes(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return [dhash(g), dhash(cv2.flip(g, 1)), dhash(cv2.flip(g, 0)), dhash(cv2.flip(g, -1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--out", default="outputs/split.json")
    ap.add_argument("--hamming", type=int, default=10, help="max dHash distance to be same group")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = list_images(args.images)
    n = len(paths)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    # 1) same Roboflow base name
    base = {}
    for i, p in enumerate(paths):
        b = os.path.basename(p).split(".rf.")[0]
        if b in base:
            union(i, base[b])
        else:
            base[b] = i

    # 2) perceptual-hash near duplicates
    H = [hashes(p) for p in paths]
    for i in range(n):
        for j in range(i + 1, n):
            d = min(int((H[i][0] != h).sum()) for h in H[j])
            if d <= args.hamming:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(stem(paths[i]))
    groups = list(groups.values())
    random.Random(args.seed).shuffle(groups)

    dev, ev = [], []
    target_dev = int(round(0.8 * n))
    for g in groups:
        (dev if len(dev) < target_dev else ev).extend(g)

    json.dump({"dev": dev, "eval": ev}, open(args.out, "w"), indent=1)
    print(f"{n} images, {len(groups)} groups -> dev={len(dev)}, eval={len(ev)}")


if __name__ == "__main__":
    main()
