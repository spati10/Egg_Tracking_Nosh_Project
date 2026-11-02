"""TASK 4 - Segment + track individual eggs in a video.

  yt-dlp -f "mp4" -o data/video.mp4 "<YOUTUBE_URL>"
  python src/step5_video_track.py --video data/video.mp4 --out outputs/video/tracked.mp4 \
         --params '{"method":"hsv","s_max":60,"v_min":140}'

Pipeline per frame: segment -> split touching eggs (watershed) -> centroids ->
Hungarian matching to existing tracks (constant-velocity prediction) -> draw ID.
"""
import argparse
import json
import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from egg_seg import segment_instances


class Tracker:
    def __init__(self, max_dist=80, max_missing=15):
        self.max_dist, self.max_missing = max_dist, max_missing
        self.tracks, self.next_id = {}, 1  # id -> dict(c, v, missing)

    def update(self, cents):
        ids = list(self.tracks)
        assigned = [None] * len(cents)
        used = set()
        if ids and cents:
            pred = np.array([self.tracks[i]["c"] + self.tracks[i]["v"] for i in ids])
            cost = np.linalg.norm(pred[:, None] - np.array(cents)[None], axis=2)
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] <= self.max_dist:
                    t = self.tracks[ids[a]]
                    newc = np.array(cents[b], float)
                    t["v"] = 0.5 * t["v"] + 0.5 * (newc - t["c"])
                    t["c"], t["missing"] = newc, 0
                    assigned[b] = ids[a]
                    used.add(ids[a])
        for i in ids:
            if i not in used:
                self.tracks[i]["missing"] += 1
        for i in [i for i in ids if self.tracks[i]["missing"] > self.max_missing]:
            del self.tracks[i]
        for b, c in enumerate(cents):
            if assigned[b] is None:
                self.tracks[self.next_id] = dict(c=np.array(c, float), v=np.zeros(2), missing=0)
                assigned[b] = self.next_id
                self.next_id += 1
        return assigned


def color_for(i):
    rng = np.random.RandomState(i * 7 + 3)
    return tuple(int(x) for x in rng.randint(60, 255, 3))


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


def is_egg_candidate(m, min_area=800, max_area=25000, min_solidity=0.80, max_aspect=1.45):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if not (min_area <= a <= max_area):
        return False

    x, y, w, h = cv2.boundingRect(c)
    aspect = w / float(h)
    # Reject elongated horizontal/vertical non-egg shapes (e.g. shirt sleeves, tables)
    if not (0.65 <= aspect <= max_aspect):
        return False

    # Exclude bottom-right corner watermark / channel logo
    if x > 500 and y > 290:
        return False

    # Solidity: real eggs are solid ovals (area / hull_area >= 0.80)
    # Fabric folds, sleeves, and shirts have jagged perimeters and lower solidity
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    if ha == 0 or (a / ha) < min_solidity:
        return False

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/video/tracked_video.mp4")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--max_dist", type=float, default=80)
    ap.add_argument("--max_missing", type=int, default=15)
    ap.add_argument("--min_area", type=int, default=800, help="ignore tiny blobs (pixels)")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    params = parse_params(args.params)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, fps, (W, H))
    trk = Tracker(args.max_dist, args.max_missing)

    pbar = tqdm(total=total if total > 0 else None, desc="Processing video frames")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break

        insts = [m for m in segment_instances(frame, params) if is_egg_candidate(m, min_area=args.min_area)]
        cents = []
        for m in insts:
            M = cv2.moments(m, binaryImage=True)
            cents.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        ids = trk.update(cents)
        for m, c, i in zip(insts, cents, ids):
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            col = color_for(i)
            cv2.drawContours(frame, cnts, -1, col, 3)
            cv2.putText(frame, f"egg {i}", (int(c[0]) - 25, int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        vw.write(frame)
        pbar.update(1)
        n += 1
    pbar.close()
    cap.release(); vw.release()
    print(f"Wrote {args.out} ({n} frames, {trk.next_id - 1} unique IDs)")



if __name__ == "__main__":
    main()
