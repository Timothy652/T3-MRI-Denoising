# T3-MRI-Denoising
# T3 Checkpoint 2 — MRI Denoising

**Student:** Timothy Ross  
**Repository:** https://github.com/Timothy652/T3-MRI-Denoising  
**Submitted version:** release tag `checkpoint2-submission`

## Project

This project compares three separately trained NAFNet denoisers:

- NGPS
- Unmasked same-coordinate adjacent-target (naive-neighbor)
- Masked-pixel blind-spot

Non-Local Means (NLM) is an additional classical control.

The project adapts the released NGPS implementation:
https://github.com/cv-cho/NGPS

Upstream source revision:
`c04331d56ad6c5c8b9ea510f77359b154db57ebe`

The upstream revision identifies reused code; the submission tag identifies
this course project's version.

## Environment

The experiments ran in Google Colab using an NVIDIA Tesla T4 GPU.

Recorded environment:
- Python 3.13.16
- PyTorch 2.14.1+cu130
- CUDA available

Dependencies include NumPy, SciPy, OpenCV, scikit-image, tqdm, PyYAML,
NiBabel, pandas, and Matplotlib.

See `environment.json` and `installed_packages.txt` in this repository
or its `logs` folder for the recorded environment and package versions.
These logs describe the execution environment, rather than a portable
cross-platform lockfile.

Training requires a compatible CUDA-enabled PyTorch installation.
The demo supports CPU execution, although it will be slower.

## Data and separation

Source:
https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html

Download each subject's **Simulated T1 image** using **MINC** format
and **gnuzip** compression.

Subject split:
- Train: 04, 05, 06, 18, 20, 38
- Validation: 41, 42
- Test: 43, 44

Each volume has shape `(181, 256, 256)` and 1 mm voxel spacing.

Reference intensities are scaled to [0,1] per volume. Gaussian noise
with sigma 0.03, 0.06, and 0.10 is added using the recorded seeds.
Raw noisy values are not clipped.

All slices, patches, and noise versions stay in their subject's split.
The same test noise realizations are used for every method.

Clean references are used only to generate noise and score frozen
predictions. They are excluded from training, guide construction,
tuning, early stopping, and model selection.

Validation subjects are reserved. Models use a fixed schedule and
final-epoch checkpoints rather than clean-reference validation selection.

## Files

The notebook contains setup, data preparation, training, evaluation,
visualization, and demonstration code.

Additional files include:
- `course_training.yaml`: training and NGPS configuration
- `subjects.json`: source and split records
- `noise_generation.json`: corruption settings and seeds
- `evaluation_protocol.json`: declared evaluation settings
- `train_course_baseline.py`: comparator training implementation
- `volume_folder.py`: modified loader preserving raw noisy values
- `subject_metrics.csv`: per-subject results
- `slice_metrics.csv`: per-slice results
- `summary_by_noise.csv`: aggregate results

Some files may be organized into `scripts`, `datasets`, `logs`,
`configs`, or `manifests` subfolders.

The Canvas ZIP also supplies the slides, execution recording,
report, and final checkpoints. Large MRI data are obtained separately
from BrainWeb.

## Working folder layout

The notebook uses this Google Drive project root:

`/content/drive/MyDrive/T3_MRI_Denoising`

Expected paths beneath that root:

- `refrence_source/`: downloaded BrainWeb `.mnc.gz` files
- `data/noisy/sigma_0.03/train/`, `validation/`, `test/`
- `data/noisy/sigma_0.06/train/`, `validation/`, `test/`
- `data/noisy/sigma_0.10/train/`, `validation/`, `test/`
- `scoring_only/references/subject43.npy`, and equivalent subject files
- `configs/course_training.yaml`
- `manifests/`
- `runs/ngps_combined/model_final.pth`
- `runs/naive_neighbor_combined/model_final.pth`
- `runs/blindspot_combined/model_final.pth`
- `results/`
- `logs/`

`refrence_source` intentionally preserves the folder spelling used
by the notebook.

The Canvas ZIP checkpoint names map to the working paths as follows:

| ZIP checkpoint | Working path under project root |
|---|---|
| `ngps_final.pth` | `runs/ngps_combined/model_final.pth` |
| `naive_neighbor_final.pth` | `runs/naive_neighbor_combined/model_final.pth` |
| `blindspot_final.pth` | `runs/blindspot_combined/model_final.pth` |

Restore these paths before running notebook cells that load checkpoints.

## Setup

Open the submitted notebook in Google Colab and select a GPU runtime.

Mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Install the supporting packages:

```python
%pip install "numpy>=1.23" "scipy>=1.9" "opencv-python>=4.7" "scikit-image>=0.20" "tqdm>=4.64" "PyYAML>=6.0" nibabel pandas matplotlib
```

Retain Colab's compatible PyTorch installation.

Obtain the upstream implementation:

```python
!git clone https://github.com/cv-cho/NGPS.git /content/NGPS
!git -C /content/NGPS checkout c04331d56ad6c5c8b9ea510f77359b154db57ebe
```

Copy the submitted modified loader to:

`/content/NGPS/datasets/volume_folder.py`

Copy the submitted comparator script to:

`/content/NGPS/scripts/train_course_baseline.py`

Alternatively, run the notebook cells that modify the loader and
create the comparator script. The source checkout alone does not
include these course modifications.

Colab's `/content` storage is temporary. Restore the source and
modifications after a runtime reset; saved Drive files persist.

## Data preparation

For a new reproduction:

1. Download the ten named BrainWeb T1 volumes.
2. Place them in the working `refrence_source` folder.
3. Run the notebook's volume-verification cell containing `splits =`.
4. Run the noise-generation cell containing `noise_records = []`.
5. Run the combined-training preparation cell containing
   `training_root = Path("/content/mri_combined")`.

The first-subject demonstration cell uses an explicit source filename.
Adjust that filename if BrainWeb assigns a different download suffix.

The preparation cells create normalized scoring references, noisy
volumes, manifests, and 18 locally staged training volumes.

Do not rerun data generation or training merely to play the demo.
When existing prepared data and checkpoints are available, restore
their paths instead.

## Training

Training pools all three noise levels.

Shared settings:
- Five epochs
- Batch size 1
- Random 128 × 128 crops
- Central fraction 0.25
- AdamW, learning rate 0.0002
- Weight decay 0.00001
- Seed 42
- 792 updates per epoch; 3,960 updates per model

After setup and staging the combined data, run from `/content/NGPS`:

```bash
python scripts/train_ngps.py \
  --data-root /content/mri_combined \
  --config /content/drive/MyDrive/T3_MRI_Denoising/configs/course_training.yaml \
  --output-dir /content/drive/MyDrive/T3_MRI_Denoising/runs/ngps_combined \
  --device cuda --central-fraction 0.25 --save-every 1
```

```bash
python scripts/train_course_baseline.py \
  --method naive_neighbor \
  --data-root /content/mri_combined \
  --config /content/drive/MyDrive/T3_MRI_Denoising/configs/course_training.yaml \
  --output-dir /content/drive/MyDrive/T3_MRI_Denoising/runs/naive_neighbor_combined \
  --device cuda --central-fraction 0.25 --save-every 1
```

```bash
python scripts/train_course_baseline.py \
  --method blindspot \
  --data-root /content/mri_combined \
  --config /content/drive/MyDrive/T3_MRI_Denoising/configs/course_training.yaml \
  --output-dir /content/drive/MyDrive/T3_MRI_Denoising/runs/blindspot_combined \
  --device cuda --central-fraction 0.25 --save-every 1
```

In Colab, shell commands can be run in a `%%bash` cell after changing
directory to `/content/NGPS`.

Expected output: epoch progress and `model_final.pth` for each method.
Original NGPS training took 18.08 minutes; blind-spot took approximately
6.4 minutes. Runtime varies with the execution environment.

## Evaluation

After restoring data and checkpoints:

1. Run notebook setup/import cells.
2. Run the checkpoint-path cell containing `checkpoints =`.
3. Run the full evaluation cell importing
   `gaussian_laplace` and `denoise_nl_means`.

The evaluation cell contains the inference and metric implementation;
it does not train models.

Expected outputs:
- `results/subject_metrics.csv`: 30 rows
- `results/slice_metrics.csv`: 5,430 rows
- `results/summary_by_noise.csv`: 15 rows
- Denoised volumes under `results/predictions/`

Evaluation processes 181 slices per subject at each noise level.

PSNR and SSIM use data range 1; SSIM uses window size 7.
HFEN is Laplacian-of-Gaussian error RMSE, sigma 1.5 pixels,
truncate 4; it is not normalized HFEN.

Denoiser outputs are clipped to [0,1]. The noisy control is unclipped.
Metrics are averaged within subjects, then equally across subjects
and noise levels.

Inference timing excludes disk I/O and metric calculation.
Neural methods took approximately 7–8 seconds per volume on GPU;
NLM took approximately 21–23 seconds on CPU. These timings are not
hardware-matched comparisons.

Run the figure cell containing `slice_index = 90` to create six
matched comparison figures under `results/figures/`.

## Demonstration without retraining

Restore the project paths, install dependencies, and load the source.
Run only the demonstration cell containing:

```python
subject = "43"
sigma = 0.10
slice_index = 90
```

The cell loads the final NGPS-trained NAFNet checkpoint, generates a new
prediction from the noisy slice, then loads the reference for scoring.

Expected approximate single-slice results:
- Noisy: PSNR 19.97 dB; SSIM 0.2789
- NGPS: PSNR 31.00 dB; SSIM 0.8368

The output displays noisy input, NGPS prediction, and reference.
These single-slice values are not the full-test averages.

The Canvas ZIP includes `execution_demo.mp4`, showing actual code
execution and generated output.

## Main results and limitations

Overall mean PSNR:
- Noisy: 24.966 dB
- Blind-spot: 30.282 dB
- NGPS: 31.765 dB
- Naive-neighbor: 32.434 dB
- NLM: 33.282 dB

The declared target required NGPS to match or exceed both trained
comparators, with zero tolerance. NGPS trailed naive-neighbor by
0.670 dB; the target was not met.

Limitations include simulated data, synthetic Gaussian noise,
two test subjects, one training seed, five training epochs,
restricted training slice locations, and incomplete improvement
and ablation experiments. Phantom results do not establish
clinical effectiveness.

## Attribution

Timothy Ross completed the project individually with AI assistance
for code and explanations. The NGPS sampler, loss functions, and
NAFNet implementation were reused from the released repository.
Original attribution and license information should be retained
with redistributed upstream source.
