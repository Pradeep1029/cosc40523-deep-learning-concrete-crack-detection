# Concrete Crack Detection — COSC 40523 / 50523 (Fall 2026)

**Team Vectorise:** Jayapradeep Jayaraman Srinivas and Harshvardhan Datre, Texas Christian University.

This repository compares (1) a 16 × 16 grayscale logistic regression baseline, (2) a scratch CNN, and (3) an ImageNet-pretrained MobileNetV2 with a frozen feature extractor on binary classification of concrete surface photographs.

> **Status (November 10, 2026):** The code and automated synthetic-image tests are complete. No official-dataset training results have been verified in this environment. Do **not** cite synthetic tests or a development subset as full-dataset scores. The IEEE manuscript in `report/` is a structurally complete draft with an explicitly unfilled empirical results section until `runs/final_seed42/metrics.json` exists.

## Dataset and provenance

Use **Çağlar Fırat Özgenel, Concrete Crack Images for Classification, version 1** ([DOI 10.17632/5y9wdsg2zt.1](https://doi.org/10.17632/5y9wdsg2zt.1), CC BY 4.0). The dataset contains 40,000 227 × 227 RGB patches: 20,000 `Positive` (crack = 1) and 20,000 `Negative` (no crack = 0). They come from 458 larger source photographs of METU campus buildings.

Download the **original dataset** from [Mendeley Data](https://data.mendeley.com/datasets/5y9wdsg2zt/1), unpack its RAR archive outside Git, and arrange its files so that the given `--dataset-root` directory contains exactly the two class folders (nested folders within them are accepted):

```
data/
  Positive/
    00001.jpg
    ...
  Negative/
    00001.jpg
    ...
```

Never upload the dataset to GitHub as part of the assignment. Follow the original CC BY 4.0 license when redistributing any images.

**Important leakage limitation:** The cropped images originate from 458 photographs. The default image-level stratified split can place patches from the **same original photo** in train and test. It measures generalization to held-out patches, *not independent photographs*. If you have independently verified source-photo IDs, create a CSV with `relative_path,source_group` for *every* patch and use `--source-groups path/to/map.csv`. Do **not** assign made-up source IDs or use arbitrary filename groups: they do not solve leakage.

## Installation

Python 3.10+ with PyTorch and torchvision. For GPU machines, install the [official PyTorch wheel](https://pytorch.org/get-started/locally/) matching your CUDA setup first.

```bash
python3 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -e '.[test]'
pytest -q
```

Alternatively: `python -m pip install -r requirements.txt` and `PYTHONPATH=src pytest -q`.

## Main experiment (3 models on the same split)

```bash
python -m crack_detection.train \
  --dataset-root data --model all --seed 42 \
  --epochs 5 --batch-size 64 --learning-rate 0.001 \
  --workers 2 --verify-images --output-dir runs/final_seed42
python scripts/make_figures.py \
  --metrics runs/final_seed42/metrics.json --output-dir report/figures
```

MobileNetV2 fetches **ImageNet-1K V1** weights on first execution. If this fails, resolve network access or use a machine where the weights are cached; `--no-pretrained` runs with a random frozen backbone and is **not** the transfer-learning experiment. On a CPU-only machine, training may be slow. Use GPU-backed Colab if available (upload/clone this repo, attach Drive-mounted data, and run the same CLI).

### Development/pilot run (not final data)

```bash
python -m crack_detection.train --dataset-root data --model all \
  --limit-per-class 100 --epochs 1 --batch-size 16 \
  --output-dir runs/pilot_seed42
```

The pilot uses only 200 total samples; its scores **must not** appear as full-dataset evidence. `--no-augmentation` allows a controlled flip-augmentation ablation. Put it in a separate output folder and use the identical seed, data, and architecture. Use `--source-groups verified.csv` if an actual source-photo mapping exists.

## Recorded protocol

| Component | Locked setup |
| --- | --- |
| Classes | `Negative` = 0; `Positive` = 1 |
| Split | Approximately 70% train / 15% val / 15% test, seed 42 |
| Split method | Image-level stratified default; source-photo grouped only with verified mapping |
| Logistic regression | 16 × 16 grayscale -> 256 pixels, `StandardScaler` fit only on train, `LogisticRegression(max_iter=1000)` |
| Scratch CNN | RGB 128 × 128; three conv/BN/ReLU blocks, first two with max pooling, global average pool, 2-class head |
| MobileNetV2 | RGB 224 × 224, ImageNet normalization, ImageNet-1K V1 frozen feature extractor and trainable 2-class head |
| Neural optimization | Adam, learning rate 0.001, cross-entropy, up to 5 epochs, batch size 64 |
| Augmentation | Horizontal and vertical flips *only* on train, not validation/test |
| Selection | Highest **validation F1** epoch; test measured once after selection |
| Determinism | Python, NumPy, PyTorch seeds; seeded shuffle, deterministic CuDNN when feasible; platform/version log |
| Reporting | Accuracy, precision, recall, F1, negative/positive confusion matrix, duration, errors, training history |

`runs/final_seed42/` contains `metrics.json`, `splits.csv`, and `<model>_errors.csv`. Metrics include platform versions, exact args, per-split class counts, training time, selected epochs, and whether a subset was used. The script intentionally **does not save checkpoints**; a fully reproducible rerun is the intended result. Workstation- and CUDA-level nondeterminism remains possible across versions.

## Figures and IEEE report

The LaTeX IEEE conference source is in `report/final_report.tex` and builds with **IEEEtran**:

```bash
cd report
pdflatex -interaction=nonstopmode final_report.tex
pdflatex -interaction=nonstopmode final_report.tex
```

The report is deliberately divided into **five pages of main content** and references on an additional page. `scripts/make_figures.py` converts genuine recorded metrics into `report/figures/` PDF plots and a LaTeX results table. Do **not** copy numeric results into the report without first validating the underlying `metrics.json` and exact dataset split. The manuscript currently labels empirical metrics as unavailable.

## Checks and limitations

Run `pytest -q`. Current tests check image split reproducibility and disjoint groups, erroneous group maps, shape of model outputs and transforms, frozen MobileNet parameter flags, metric computations, dataset verification, and an end-to-end logistic smoke run on **synthetic** images. These tests establish code behavior, **not concrete-dataset accuracy**. Check the official image counts, unreadable images, and potential near-duplicates before interpreting any final result.

## References

1. Özgenel, Ç. F. (2018), *Concrete Crack Images for Classification*, Mendeley Data V1. [DOI](https://doi.org/10.17632/5y9wdsg2zt.1)
2. Özgenel, Ç. F., and Gönenç Sorguç, A. (2018), "Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings," ISARC 2018. [DOI](https://doi.org/10.22260/ISARC2018/0094)
3. Zhang, L., Yang, F., Zhang, Y. D., and Zhu, Y. J. (2016), "Road Crack Detection Using Deep Convolutional Neural Network," ICIP. [DOI](https://doi.org/10.1109/ICIP.2016.7533052)
4. Sandler, M. et al. (2018), "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR. [DOI](https://doi.org/10.1109/CVPR.2018.00474)
