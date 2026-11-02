"""Shared helpers: file listing, mask IO, metrics, overlays."""
import os
import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder):
    files = []
    for root, _, fs in os.walk(folder):
        for f in fs:
            if f.lower().endswith(IMG_EXT):
                files.append(os.path.join(root, f))
    return sorted(files)


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 127)


def metrics(gt, pred):
    """Binary segmentation metrics. gt/pred: bool arrays of equal shape."""
    gt, pred = gt.astype(bool), pred.astype(bool)
    tp = float((gt & pred).sum())
    fp = float((~gt & pred).sum())
    fn = float((gt & ~pred).sum())
    tn = float((~gt & ~pred).sum())
    eps = 1e-9
    if tp + fp + fn == 0:  # both empty
        return dict(iou=1.0, dice=1.0, precision=1.0, recall=1.0, pixel_acc=1.0)
    return dict(
        iou=tp / (tp + fp + fn + eps),
        dice=2 * tp / (2 * tp + fp + fn + eps),
        precision=tp / (tp + fp + eps),
        recall=tp / (tp + fn + eps),
        pixel_acc=(tp + tn) / (tp + tn + fp + fn),
    )


def overlay(img, mask, color=(0, 0, 255), alpha=0.55):
    out = img.copy()
    col = np.zeros_like(img)
    col[:] = color
    m = mask.astype(bool)
    out[m] = cv2.addWeighted(img, 1 - alpha, col, alpha, 0)[m]
    return out
