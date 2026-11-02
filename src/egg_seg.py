"""TASK 2b - Classical egg segmentation (no deep learning).

segment(img, params)            -> binary mask (uint8 0/255), same size as img
segment_instances(img, params)  -> list of per-egg masks (used for video tracking)

Everything is controlled by a params dict so each experiment = one param change.
"""
import cv2
import numpy as np

DEFAULT = dict(
    method="hsv",       # 'hsv' | 'lab_otsu' | 'grabcut'
    work_size=512,      # resize longest side for speed / consistent kernel sizes
    blur=5,             # gaussian kernel (odd), 0/1 = off
    # hsv
    s_max=60, v_min=140,
    # lab_otsu
    channel="a", invert=True,
    # grabcut
    gc_margin=0.15, gc_iters=5,
    # post-processing
    open_k=5, close_k=9,
    min_area=0.02, max_area=0.70,   # fraction of image
    keep_largest=1,                 # 0 = keep all components (video)
    fill_holes=True, hull=False,
    # instance splitting (video)
    split_frac=0.6,
)


def _kernel(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _raw_mask(img, p):
    b = cv2.GaussianBlur(img, (p["blur"], p["blur"]), 0) if p["blur"] > 1 else img
    m = p["method"]
    if m == "hsv":
        hsv = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)
        return (((hsv[..., 1] <= p["s_max"]) & (hsv[..., 2] >= p["v_min"])) * 255).astype(np.uint8)
    if m == "lab_otsu":
        lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
        ch = {"L": 0, "a": 1, "b": 2}[p["channel"]]
        _, bw = cv2.threshold(lab[..., ch], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(bw) if p["invert"] else bw
    if m == "grabcut":
        h, w = b.shape[:2]
        g = p["gc_margin"]
        rect = (int(w * g), int(h * g), int(w * (1 - 2 * g)), int(h * (1 - 2 * g)))
        gc = np.zeros((h, w), np.uint8)
        cv2.grabCut(b, gc, rect, np.zeros((1, 65)), np.zeros((1, 65)), p["gc_iters"], cv2.GC_INIT_WITH_RECT)
        return ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    if m == "bg_contrast":
        h, w = b.shape[:2]
        bm = p.get("border_margin", 0.08)
        # Define outer border region (8% margin from all 4 edges)
        border_mask = np.ones((h, w), bool)
        border_mask[int(h * bm):int(h * (1 - bm)), int(w * bm):int(w * (1 - bm))] = False
        # Convert to CIELAB space and find average background color
        lab = cv2.cvtColor(b, cv2.COLOR_BGR2LAB).astype(float)
        bg_mean = lab[border_mask].mean(axis=0)
        # Calculate Euclidean color distance (Delta-E) from background for every pixel
        dist = np.linalg.norm(lab - bg_mean, axis=2)
        dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Otsu's thresholding automatically separates foreground from background
        _, bw = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw
    raise ValueError(f"unknown method {m}")


def _post(mask, p):
    if p["open_k"] > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(p["open_k"]))
    if p["close_k"] > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(p["close_k"]))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    total = mask.size
    comps = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
             if p["min_area"] * total <= stats[i, cv2.CC_STAT_AREA] <= p["max_area"] * total]
    comps.sort(reverse=True)
    if p["keep_largest"] > 0:
        comps = comps[: p["keep_largest"]]
    out = np.zeros_like(mask)
    for _, i in comps:
        out[lab == i] = 255
    if p["fill_holes"] or p["hull"]:
        cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(mask)
        for c in cnts:
            if p["hull"]:
                c = cv2.convexHull(c)
            cv2.drawContours(out, [c], -1, 255, cv2.FILLED)
    return out


def segment(img, params=None):
    p = {**DEFAULT, **(params or {})}
    H, W = img.shape[:2]
    s = p["work_size"] / max(H, W)
    small = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    mask = _post(_raw_mask(small, p), p)
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    return mask


def split_instances(mask, frac=0.6):
    """Split touching eggs with distance-transform + watershed. Returns list of masks."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents(m)
    markers = np.zeros(m.shape, np.int32)
    nxt = 1
    for i in range(1, n):
        d = dist * (lab == i)
        sure = (d > frac * d.max()).astype(np.uint8)
        k, sl = cv2.connectedComponents(sure)
        for j in range(1, k):
            markers[sl == j] = nxt
            nxt += 1
    bg = nxt
    markers[m == 0] = bg
    dn = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ws = cv2.watershed(cv2.cvtColor(255 - dn, cv2.COLOR_GRAY2BGR), markers)
    out = []
    for k in range(1, bg):
        inst = ((ws == k) & (m > 0)).astype(np.uint8) * 255
        if inst.sum() > 0:
            out.append(inst)
    return out


def segment_instances(img, params=None):
    p = {**DEFAULT, "keep_largest": 0, **(params or {})}
    return split_instances(segment(img, p), p["split_frac"])
