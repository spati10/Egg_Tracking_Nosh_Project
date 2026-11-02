# Task 2: Classical Segmentation and Experiment Log

Dataset: Egg Defect and Quality Dataset, 160 images.

The goal here was to build a segmentation pipeline using only classical image processing (no deep learning) and get it as close as I could to the SAM masks from Task 1, which I treat as ground truth.

## 1. Splitting the data without leaking

I did the split before writing any segmentation code, because a plain random split would have been misleading for this dataset. There were two sources of leakage I had to deal with.

The first is Roboflow augmentation. The export contains several augmented copies of the same source photo, named like not_damaged_10_jpg.rf.<hash>.jpg. If one copy ends up in dev and another in eval, the eval score is inflated because the model has effectively seen that image already.

The second is near-duplicate photos. Some images were clearly shot a few seconds apart from the same angle, with only small changes in lighting or rotation.

To handle both, I wrote src/step2_split.py. It does three things:

1. It groups every .rf.<hash> variant of the same base image into one cluster.
2. It computes an 8x8 difference hash (dHash) for each image in four orientations (original, horizontal flip, vertical flip, and diagonal flip). Any two images with a Hamming distance of 10 or less get merged into the same group.
3. It assigns whole groups to either dev or eval at a 4:1 ratio, with a fixed seed of 42.

This gave 128 images in dev and 32 in eval, and I found no overlap between the two sets.

## 2. Why this is harder than it looks

At first glance this seems like a simple thresholding job. After looking through the images, a few things got in the way:

- The eggs are not all the same color. Some are bright white and some are tan or brown, so a rule tuned for white eggs misses the brown ones entirely.
- Many tables are light colored, off-white, or wooden. White eggs blend into white tables, and brown eggs blend into wood.
- Overhead lighting puts bright specular patches on the shell. With simple thresholding these blow out and leave holes or ragged gaps inside the egg mask.
- Real eggs are smooth, solid and convex. A mask with holes or sharp concave notches is physically wrong, which gave me something to build on.

## 3. Experiment log

All trials below were run on the dev split. That is 126 images, since two of them had blank SAM masks and were skipped. Results are logged in experiments/experiment_log.csv.

### Trial 1: HSV baseline (E01_baseline_hsv)

Idea: eggs should be bright and not very colorful, so I kept pixels with saturation up to 60 and value of at least 140.

Result: mean IoU 0.2991, median IoU 0.0011, precision 0.3514, recall 0.4096.

The mean looks okay, but the median of 0.0011 shows that it failed on most images. On light tables, the table itself passed the filter, so keeping the largest blob picked the whole background instead of the egg. On brown eggs it found nothing. Discarded.

### Trial 2: CIELAB a-channel with Otsu (E02_lab_otsu_a)

Idea: CIELAB keeps lightness separate from color (a is green to red, b is blue to yellow), so I tried Otsu on the a-channel to separate shell color from neutral tables without hand-picked cutoffs.

Result: mean IoU 0.1390, median IoU 0.0001, precision 0.1758, recall 0.3099.

This was a clear failure. Both the shell and the background sit close to neutral on the a-channel (around 128), so the histogram has no clear two-peak shape. My guess is that Otsu was just splitting on noise. Discarded.

### Trial 3: CIELAB L-channel with Otsu (E03_lab_otsu_L)

Idea: eggs usually reflect more diffuse light than the surface they sit on, so Otsu on the lightness channel should find a sensible threshold for each photo separately.

Result: mean IoU 0.3276, median IoU 0.1631, precision 0.3817, recall 0.4570.

This was the first real improvement, especially in the median, which went from near zero to 0.1631. The threshold adapted well to different exposures. The problems that remained were highlights and shadows on the egg, which left gaps along the boundary. I kept it as the base for later trials.

### Trial 4: Closing plus convex hull (E04_lab_otsu_L_hull)

Idea: since an egg is convex, internal holes and concave notches in the mask are errors. I added a morphological closing (kernel 11) and then took the convex hull of the largest contour.

Result: mean IoU 0.3285, median IoU 0.2217, precision 0.3688, recall 0.5400.

Recall went up by roughly 10 points (0.457 to 0.540) and the median rose to 0.2217, because the hull fills the highlight holes. It did not fix the case where the table is brighter than the egg, since Otsu on L still flips foreground and background there. I kept the hull step.

### Trial 5: Background contrast from the border plus hull (E05_bg_contrast_hull)

Idea: instead of guessing whether the egg is brighter or darker than the table, I used a spatial assumption. The egg is near the center, and the outer 8% border of the image is almost certainly table.

How it works:

1. Take the outer 8% margin of the image in CIELAB and average it to get a background color (L, a, b).
2. For every pixel, compute the Euclidean distance in Lab from that background color.
3. Run Otsu on the normalized distance map. An egg should differ strongly from the border color whether it is white, cream or brown.
4. Clean up with an opening (kernel 5) to remove speckles, a closing (kernel 11) to merge broken edges, and then the convex hull for the final shape.

Result: mean IoU 0.3494, median IoU 0.2987, precision 0.4784, recall 0.4941, about 0.037 seconds per image.

This had the best balance of everything I tried. Precision and recall are close to each other at around 48 to 49 percent, and it has the highest median IoU on dev. I chose it as the final algorithm.

## 4. Summary table

| Run | Method | Split | Images | Mean IoU | Median IoU | Mean Dice | Precision | Recall | Sec/img | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| E01 | HSV (S <= 60, V >= 140) | dev | 126 | 0.2991 | 0.0011 | 0.3537 | 0.3514 | 0.4096 | 0.018 | Discarded. Fails on brown eggs and light tables. |
| E02 | Lab a-channel + Otsu | dev | 126 | 0.1390 | 0.0001 | 0.1820 | 0.1758 | 0.3099 | 0.015 | Discarded. Not enough contrast on neutral backgrounds. |
| E03 | Lab L-channel + Otsu | dev | 126 | 0.3276 | 0.1631 | 0.3881 | 0.3817 | 0.4570 | 0.015 | Kept as a base. Handles exposure differences. |
| E04 | Lab L + Otsu + convex hull | dev | 126 | 0.3285 | 0.2217 | 0.3960 | 0.3688 | 0.5400 | 0.016 | Kept the hull. Fixes highlight holes. |
| E05 | Border background contrast + hull | dev | 126 | 0.3494 | 0.2987 | 0.4319 | 0.4784 | 0.4941 | 0.037 | Chosen. Works on white and brown eggs. |
| FINAL | E05 on held-out eval | eval | 31 | 0.2825 | 0.0821 | 0.3510 | 0.4024 | 0.4015 | 0.095 | Final benchmark. |

## 5. Final check on the eval split

I did not touch the eval split while experimenting. After fixing E05 as the final method, I ran it once with the --final flag:

```bash
python src/step3_run_experiment.py --name FINAL \
    --desc "Best algorithm: Border Delta-E distance + Otsu + Convex Hull" \
    --params '{"method":"bg_contrast","hull":true,"open_k":5,"close_k":11}' \
    --split eval --final
```

This ran on 31 images (one was skipped because its ground truth mask was empty). The mean IoU was 0.2825, precision was 0.4024 and recall was 0.4015. The predicted masks are saved in outputs/pred_final/ so I can use them for the error analysis in Task 3.

One thing worth being upfront about: the eval numbers are lower than dev. Mean IoU dropped from 0.3494 to 0.2825, and the median fell a lot more, from 0.2987 to 0.0821. That means a good share of the eval images are failing badly, probably where the border assumption breaks or the egg touches the edge of the frame. I plan to look at exactly those images in Task 3 rather than claim the method generalizes well. With only 31 eval images, a handful of total failures also moves these numbers quite a bit.