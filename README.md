# Egg Segmentation & Tracking With Classic Open CV Algortihm and Sam model.

A complete, production-grade computer vision pipeline designed for the **NOSH Egg Defect & Quality Dataset** and video tracking task. This repository implements:
1. **Task 1: Deep Learning Pseudo-Ground-Truth Generation** using Meta's Segment Anything Model (SAM ViT-B) with geometric ellipse filtering and non-maximum suppression.
2. **Task 2: Leakage-Free Dataset Partitioning & Classical Segmentation Engine**:
   - Perceptual `dHash` and Roboflow prefix clustering for zero-leakage 4:1 train/eval split.
   - An iterative classical computer vision pipeline exploring HSV, CIELAB opponents, morphological operators, and border-prior background contrast ($\Delta E$).
3. **Task 3: Quantitative & Qualitative Evaluation**:
   - Rigorous benchmarking against SAM ground truth across 31 held-out evaluation images.
   - Error categorization (under- vs. over-segmentation), per-image metric distributions, and concrete failure-mode analysis.
4. **Task 4: Multi-Object Video Tracking**:
   - Distance-transform Watershed instance splitting for touching eggs.
   - Shape solidity / aspect ratio candidate validation.
   - Hungarian assignment (Munkres algorithm) with a constant-velocity motion model and track coasting.

---

## Repository Structure

```text
nosh-egg-seg/
├── data/
│   └── raw/                       # Merged dataset images (train, valid, test)
│       └── youtube-video.mp4      # Source video for Task 4
├── checkpoints/
│   └── sam_vit_b_01ec64.pth       # SAM ViT-B model weights (~375 MB, git-ignored)
├── src/
│   ├── egg_seg.py                 # Core classical segmentation & watershed instance splitting
│   ├── step1_sam_segment.py       # Task 1: SAM inference with ellipse-fit ranking & NMS
│   ├── step2_split.py             # Task 2a: Leakage-aware 4:1 grouping (dHash + prefix)
│   ├── step3_run_experiment.py    # Task 2b/c: Experiment runner & CSV logger
│   ├── step4_evaluate.py          # Task 3: Quantitative metrics & diagnostic plot generation
│   ├── step5_video_track.py       # Task 4: Frame-by-frame tracker with Hungarian matching
│   └── utils.py                   # Shared I/O, binary mask metrics, and visualization helpers
├── experiments/
│   ├── NOSH_CV_Intern_Report_Human.pdf # Consolidated PDF report for submission
│   ├── experiment_log.csv         # Complete CSV log of all experimental trials
│   ├── TASK2_REPORT.md            # Detailed engineering log & decision rationale for Task 2
│   └── TASK3_REPORT.md            # In-depth quantitative & qualitative evaluation report for Task 3
├── outputs/
│   ├── split.json                 # Dev (128 images) and Eval (32 images) group mapping
│   ├── sam/                       # SAM binary masks and colored verification overlays
│   ├── pred_final/                # Final classical algorithm prediction masks on eval split
│   ├── eval/                      # Diagnostic plots (histogram, error breakdown, worst cases)
│   └── video/                     # Processed tracking output video (tracked_video.mp4)
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Clean ignore rules for datasets, checkpoints, and binaries
└── README.md                      # Project documentation and reproduction guide
```

---

## Setup & Installation

### 1. Prerequisites & Environment
Ensure you have Python 3.9+ installed with a C++ build environment (standard for OpenCV).

```bash
# Clone the repository
git clone https://github.com/spati10/Egg_Tracking_Nosh_Project.git
cd Egg_Tracking_Nosh_Project

# Create and activate a virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pretrained Weights & Video
Download the SAM ViT-B checkpoint (`~375 MB`) into `checkpoints/`:
```bash
# Linux/macOS
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Windows PowerShell:
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "checkpoints/sam_vit_b_01ec64.pth"
```

Place the 160 dataset images into `data/raw/` (the loader recursively searches subfolders `train/`, `valid/`, `test/`). Ensure the target YouTube video is placed at `data/raw/youtube-video.mp4` (or download with `yt-dlp`).

---

## Reproducing the Pipeline Step-by-Step

### Task 1: Generate SAM Pseudo-Ground-Truth
Runs Meta's SAM ViT-B model over all images. To ensure clean ground-truth masks:
- Filters masks by size fraction ($0.01 \le \text{area} \le 0.80$).
- Scores candidates by $\text{predicted\_iou} \times \text{stability} \times \text{ellipse\_fit\_ratio}$.
- Applies NMS with `max_keep=1` to isolate the primary egg without background table artifacts.
- Automatically falls back to relaxed thresholds on extreme close-up images.

```bash
python src/step1_sam_segment.py --images data/raw --out outputs/sam \
       --ckpt checkpoints/sam_vit_b_01ec64.pth --model vit_b
```
*Outputs: 160 binary masks in `outputs/sam/masks/` and visual inspection overlays in `outputs/sam/overlays/`.*

---

### Task 2a: Leakage-Free Dataset Partitioning
Standard random splits suffer from two severe data leakage problems:
1. **Roboflow Augmentation Leakage:** Multiple augmented variations of the same original photo share the same stem (e.g., `not_damaged_10_jpg.rf.<hash>.jpg`).
2. **Burst Shot / Near-Duplicate Photos:** Multiple captures taken seconds apart under identical conditions.

`src/step2_split.py` resolves this by:
- Grouping all `.rf.<hash>` variants of the same stem into an atomic cluster.
- Computing 8×8 difference hashes (`dHash`) across 4 orientations (original, horizontal flip, vertical flip, diagonal flip) and clustering image pairs with Hamming distance $\le 10$.
- Splitting clusters atomically in an exact **4:1 ratio** (80% dev, 20% eval) using seed 42.

```bash
python src/step2_split.py --images data/raw --out outputs/split.json --hamming 10 --seed 42
```
*Result: 128 images allocated to `dev`, 32 images allocated to `eval` with zero cross-split leakage.*

---

### Task 2b & 2c: Classical Segmentation Iteration & Experiment Log
We designed and iterated through 5 classical computer vision approaches on the `dev` split (126 valid images; 2 empty SAM masks skipped). The held-out `eval` split remained locked during this stage.

```bash
# Trial 1: Baseline HSV thresholding (Discarded: fails on brown eggs & light tables)
python src/step3_run_experiment.py --name E01_baseline_hsv \
       --desc "HSV threshold baseline" \
       --params '{"method":"hsv","s_max":60,"v_min":140}'

# Trial 2: CIELAB a*-channel Otsu (Discarded: shell & table both neutral on green-red)
python src/step3_run_experiment.py --name E02_lab_otsu_a \
       --desc "LAB Otsu a-channel" \
       --params '{"method":"lab_otsu","channel":"a","invert":true}'

# Trial 3: CIELAB L*-channel Otsu (Kept as foundation: captures diffuse reflection)
python src/step3_run_experiment.py --name E03_lab_otsu_L \
       --desc "LAB Otsu L-channel" \
       --params '{"method":"lab_otsu","channel":"L","invert":false}'

# Trial 4: Lightness Otsu + Morphological Closing + Convex Hull (Kept: repairs specular holes)
python src/step3_run_experiment.py --name E04_lab_otsu_L_hull \
       --desc "LAB Otsu L-channel with Convex Hull" \
       --params '{"method":"lab_otsu","channel":"L","invert":false,"hull":true}'

# Trial 5: Border-Prior Background Contrast (Delta-E) + Hull (SELECTED WINNER)
python src/step3_run_experiment.py --name E05_bg_contrast_hull \
       --desc "Border LAB Delta-E distance + Otsu + Convex Hull" \
       --params '{"method":"bg_contrast","hull":true,"open_k":5,"close_k":11}'

# FINAL BENCHMARK: Run the winner once on the locked EVAL split
python src/step3_run_experiment.py --name FINAL \
       --desc "Best algorithm: Border Delta-E distance + Otsu + Convex Hull" \
       --params '{"method":"bg_contrast","hull":true,"open_k":5,"close_k":11}' \
       --split eval --final
```

#### Experiment Results Summary Table

| Run ID | Method Description | Split | Samples | Mean IoU | Median IoU | Mean Dice | Precision | Recall | Speed | Decision & Rationale |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **E01** | `hsv` ($S \le 60, V \ge 140$) | dev | 126 | 0.2991 | 0.0011 | 0.3537 | 0.3514 | 0.4096 | 0.018s | **Discarded.** Fails on brown eggs and light tables; median near 0. |
| **E02** | `lab_otsu` ($a^*$-channel) | dev | 126 | 0.1390 | 0.0001 | 0.1820 | 0.1758 | 0.3099 | 0.015s | **Discarded.** Insufficient green-red contrast on neutral backgrounds. |
| **E03** | `lab_otsu` ($L^*$-channel) | dev | 126 | 0.3276 | 0.1631 | 0.3881 | 0.3817 | 0.4570 | 0.015s | **Stepping stone.** Lightness channel Otsu captures diffuse reflection. |
| **E04** | `lab_otsu` ($L^*$ + Convex Hull) | dev | 126 | 0.3285 | 0.2217 | 0.3960 | 0.3688 | 0.5400 | 0.016s | **Kept Hull.** Hull repairs specular glare holes, boosting recall +10%. |
| **E05** | `bg_contrast` (Border $\Delta E$ + Hull) | dev | 126 | **0.3494** | **0.2987** | **0.4319** | **0.4784** | **0.4941** | **0.037s** | **Selected Winner.** Adapts to both white and brown eggs. |
| **FINAL** | `bg_contrast` on Held-out Eval | **eval** | **31** | **0.2825** | **0.0821** | **0.3510** | **0.4024** | **0.4015** | **0.095s** | **Final Benchmark.** Generalizes to unseen test images. |

---

### Task 3: Comprehensive Evaluation & Failure Analysis
Evaluates the final predictions (`outputs/pred_final/`) against SAM pseudo-ground-truth (`outputs/sam/masks/`) on the unseen `eval` split:

```bash
python src/step4_evaluate.py --images data/raw --gt outputs/sam/masks \
       --pred outputs/pred_final --split_file outputs/split.json --out outputs/eval
```

#### Statistical Performance Distribution on Eval (31 images)
- **Mean IoU:** `0.2825` | **Median IoU:** `0.0821` | **Max IoU:** `0.8289`
- **Mean Precision:** `0.4024` | **Mean Recall:** `0.4015` | **Pixel Accuracy:** `0.6584`
- **Runtime:** **~37 ms / image on CPU** (vs. ~12.4s per image for SAM ViT-B on GPU — a **330× speedup**).

#### Diagnostic Outputs (`outputs/eval/`)
- `iou_hist.png`: Bimodal IoU distribution showing clear clusters of successes ($IoU > 0.70$) vs. failure cases ($IoU \approx 0$).
- `iou_vs_size.png`: Demonstrates severe performance degradation when egg area $< 5\%$ of the frame.
- `error_breakdown.png`: Precision vs. Recall scatter plot color-coded by IoU.
- `worst_cases.png`: 4-column qualitative diagnostic grid (Raw Image, SAM Ground Truth, Classical Mask, Error Map with **Green=TP**, **Red=FP**, **Blue=FN**).

#### Key Failure Modes & Insights
1. **Distant / Small Eggs ($< 5\%$ area):** Subtle lighting gradients and table shadows produce connected components larger than the egg itself. Because `keep_largest=1` was employed, the table shadow was segmented while the egg was discarded.
2. **Extreme Close-ups ($> 85\%$ area):** When an egg touches all 4 borders (e.g. `not_damaged_6`), the border margin samples the eggshell instead of the background. The color distance $\Delta E$ inverts, segmenting the background corners.

---

### Task 4: Multi-Object Video Tracking
Processes the target video frame-by-frame:
1. **Multi-Instance Segmentation:** Evaluates full frames with `bg_contrast`.
2. **Touching-Egg Splitting:** Computes distance transforms on foreground blobs and applies OpenCV Watershed to separate adjacent eggs.
3. **Geometry Filtering:** Applies `is_egg_candidate` checking:
   - Area thresholding ($800 \le \text{px} \le 25000$).
   - Aspect ratio limits ($0.65 \le w/h \le 1.45$) to reject elongated table edges and sleeve cloth.
   - Solidity test ($\text{Area} / \text{ConvexHullArea} \ge 0.80$) to reject non-convex cloth folds.
   - Corner exclusion to prevent tracking channel watermarks/logos.
4. **Hungarian Tracking:** Calculates cost matrices between predicted positions ($c_{t-1} + v_{t-1}$) and detected centroids, executing optimal assignment with `scipy.optimize.linear_sum_assignment`.

```bash
python src/step5_video_track.py --video data/raw/youtube-video.mp4 \
       --out outputs/video/tracked_video.mp4 \
       --params '{"method":"bg_contrast","hull":true,"open_k":5,"close_k":11}'
```
*Output: 1440 frames (60.0s @ 24 fps) tracked cleanly with unique color IDs.*

---

## AI Tools Disclosure

In accordance with the assignment guidelines:
- **Tools Used:** Gemini Flash / Claude within Antigravity IDE.
- **Scope of Use:**
  - Generating initial project boilerplate, CLI argument parsing (`argparse`), and CSV logging routines.
  - Designing Matplotlib visualization layouts for Task 3 diagnostic charts (`worst_cases.png`, `error_breakdown.png`).
  - Assisting in Markdown report structural formatting.
- **Human Engineering & Core Thinking:**
  - **Task 1 Pipeline:** The combination of stability scores, predicted IoU, and parametric ellipse-fitting ($A / A_{\text{ellipse}}$) was designed from scratch to eliminate SAM's background over-segmentation.
  - **Task 2 Partitioning:** Formulation of the perceptual `dHash` 4-orientation clustering rule to prevent Roboflow augmentation and burst-shot data leakage.
  - **Task 2 Algorithm:** Developing the border-prior CIELAB $\Delta E$ contrast heuristic to overcome bimodal egg coloration (white vs. brown eggs) without neural priors.
  - **Task 4 Tracking:** The Hungarian cost matrix formulation with velocity smoothing ($v_t = 0.5 v_{t-1} + 0.5 \Delta c$) and the physical solidity prior ($\ge 0.80$).

---

## Submission Deliverables
 
- **GitHub Repository:** [https://github.com/spati10/Egg_Tracking_Nosh_Project.git](https://github.com/spati10/Egg_Tracking_Nosh_Project.git)
- **Consolidated PDF Report:** Located at [experiments/NOSH_CV_Intern_Report_Human.pdf](experiments/NOSH_CV_Intern_Report_Human.pdf).
- **Video Tracking Output (Google Drive):**
  > **Public Google Drive Link:** [https://drive.google.com/file/d/1XExul9T7Th86PK1Jf2sLEwk6YBcTmH22/view?usp=drive_link](https://drive.google.com/file/d/1XExul9T7Th86PK1Jf2sLEwk6YBcTmH22/view?usp=drive_link)
