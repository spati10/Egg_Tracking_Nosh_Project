# Task 3: Evaluation and Failure Analysis

Split: held-out eval set (31 images evaluated, 1 skipped because its ground truth mask was blank)
Ground truth: SAM masks from Task 1
Prediction: final classical algorithm (E05_bg_contrast_hull), which uses the border-based CIELAB color distance, Otsu thresholding, morphological cleanup and a convex hull.

## 1. Quantitative results

In this task I compare my classical pipeline against the SAM masks on the eval split, which I did not touch while tuning. SAM has learned what objects look like, while my method only looks at color differences, so the comparison shows fairly clearly where hand-built rules work and where they fall apart.

| Metric | Mean | Median | 25th percentile | 75th percentile | Max |
| :--- | :---: | :---: | :---: | :---: | :---: |
| IoU | 0.2825 | 0.0821 | 0.0000 | 0.5763 | 0.8289 |
| Dice | 0.3510 | 0.1517 | 0.0000 | 0.7312 | 0.9065 |
| Precision | 0.4024 | 0.2610 | 0.0000 | 0.8875 | 1.0000 |
| Recall | 0.4015 | 0.5219 | 0.0000 | 0.7643 | 1.0000 |
| Pixel accuracy | 0.6584 | 0.6847 | 0.5145 | 0.8433 | 0.9495 |

A few things stand out from these numbers:

- The pipeline never returned an empty mask. It produced something on every image, though not always the egg.
- About a third of the images (32.3 percent) reached an IoU above 0.50, and the best one hit 0.8289 with precision of 1.0.
- The gap between the mean (0.2825) and the median (0.0821) shows that results are split. Some images work well and a good number fail completely.
- It runs in roughly 35 to 95 ms per image on CPU. SAM ViT-B took about 12.4 seconds per image on my GTX 1650, so the classical method is somewhere between 130 and 330 times faster.

## 2. Qualitative analysis

The plots are saved in outputs/eval/:

- iou_hist.png shows the IoU distribution. It has one cluster of good results and one cluster of failures.
- iou_vs_size.png and iou_vs_brightness.png plot IoU against egg size and scene brightness.
- error_breakdown.png plots precision against recall, which helps separate over-segmentation from under-segmentation.
- worst_cases.png is a grid of the input, the SAM mask, my prediction and an error map. In the error map, green is a correct pixel, red is a false positive and blue is a false negative.

### What worked

The best results (IoU between 0.70 and 0.83) came from normal tabletop photos where the egg takes up roughly 15 to 50 percent of the frame. Examples are not_damaged_130 (0.8289), not_damaged_140 (0.8072), not_damaged_90 (0.7862) and not_damaged_153 (0.7328). Three things seem to explain this:

1. The 8 percent border really did contain just the table, so the background color estimate was accurate.
2. The convex hull filled in the glare holes on the shell, and the final outline ended up very close to SAM's.
3. Precision on these images was 98 to 100 percent, so there was almost nothing outside the egg being picked up.

### What failed

Going through per_image_metrics.csv and worst_cases.png, I found two main failure patterns.

Failure 1: small, far-away eggs. The images with an IoU of zero are not_damaged_118, not_damaged_97, not_damaged_27, not_damaged_49 and not_damaged_9. In all of them the egg covers only about 1.2 to 5.6 percent of the frame (not_damaged_27 is 0.0123). In these images, gradual lighting changes or shadows on the table form a connected region that is bigger than the egg. Because I keep only the largest component, the algorithm picked the shadow and dropped the egg. The area limits I set (min_area of 0.02) and the largest-blob rule both favor medium and large objects, which is a poor fit for these cases.

Failure 2: extreme close-ups. The clearest example is not_damaged_6, where the egg covers about 87 percent of the image and the IoU is only 0.0498. The egg runs past the frame on every side, so the outer border is eggshell instead of table. My method then used the egg as its background color, and the result was inverted: it segmented the small dark corners of the table instead of the egg. The idea that the border is always background works when the egg sits inside the frame, but it breaks completely once the egg touches all four edges. I only saw this clearly in one image, so I can't say how common it is in general.

## 3. How it compares with SAM

| Aspect | SAM (ViT-B) | My classical pipeline |
| :--- | :--- | :--- |
| Understanding | Has learned what an object is, so it doesn't depend on a particular color. | Uses only low-level color contrast against the image border. |
| Lighting | Handles shadows, glare and exposure changes much better. | Struggles with strong shadows and reflective tables. |
| Scale | Finds eggs whether they fill 1 percent or 90 percent of the frame. | Works best on typical framing (about 10 to 60 percent) and fails on tiny or extreme close-up eggs. |
| Speed and hardware | Needs a GPU and took about 12.4 s per image on a GTX 1650. | About 37 ms on CPU with no GPU. |
| Transparency | Hard to inspect or explain. | Every step is simple and easy to tune. |

## 4. What I would improve

If I kept working on this, these are the three changes I would try first:

1. Check the border before trusting it. If the border pixels are very uniform in a way that suggests they are eggshell (for example, very low variance), the sampling window could be changed or the method could fall back to another way of estimating the background. This targets the close-up failure.

2. Pick the egg by shape, not by size. Instead of keeping the largest blob, give each candidate a score that combines its area, how well an ellipse fits it, and how circular it is. A small but clearly egg-shaped region would then win over a big, shapeless shadow. This targets the small-egg failure.

3. Try superpixels or GrabCut. Grouping pixels into color-consistent regions (for example with SLIC) before thresholding might keep the egg edge sharp while ignoring slow lighting changes across the table. I haven't tested this, so I'm treating it as an idea and not a promised fix.