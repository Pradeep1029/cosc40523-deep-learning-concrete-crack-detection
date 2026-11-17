# Concrete crack detection — COSC 50523

Fall 2026, TCU  
Jayapradeep Jayaraman Srinivas and Harshvardhan Datre (Team Vectorise)

This is our deep learning project. We trained three classifiers to predict whether a photo patch of concrete contains a crack:

- Logistic regression on 16×16 grayscale pixels.
- A small CNN trained from scratch on 128×128 RGB images.
- MobileNetV2, starting from ImageNet weights, with its feature extractor frozen and a new two-class head.

The code, report, and real results are here. We did **not** include the dataset or model checkpoints.

## Results

We evaluated all three models on the same 6,000 held-out patches from the 40,000-image dataset. The random seed was 42.

| Model | Accuracy | Precision | Recall | F1 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 90.50% | 93.86% | 86.67% | 0.9012 | 170 | 400 |
| Our CNN | 99.73% | 99.80% | 99.67% | 0.9973 | 6 | 10 |
| MobileNetV2 | 99.90% | 99.80% | 100.00% | 0.9990 | 6 | 0 |

Those are the results from our actual training runs, not numbers taken from another paper. The [successful GitHub Actions run](https://github.com/Pradeep1029/cosc40523-deep-learning-concrete-crack-detection/actions/runs/42958596193) produced the metrics and report.

**Important limitation:** This dataset was cut from 458 original photos. We had no verified mapping from each patch back to its parent photo, so our random train/validation/test split can put different crops of the same photo into different sets. These scores are for *held-out patches*. They do not tell us how well the models would work on photographs of a new building. We discuss this in the report.

## Dataset

We used Ç. F. Özgenel's [Concrete Crack Images for Classification](https://doi.org/10.17632/5y9wdsg2zt.1), version 1 (CC BY 4.0): 20,000 cracked and 20,000 non-cracked 227×227 RGB images.

The original source is Mendeley Data. For the completed runs we downloaded a [public Kaggle mirror](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification), checked the class counts, and opened all 40,000 files to confirm their dimensions. The downloaded archive's SHA-256 is recorded in `results/dataset_provenance.json`.

To get the same data, run:

```bash
python scripts/download_dataset.py --output data --source auto
```

The downloader tries the Kaggle mirror first and then another public mirror. You can also download the original release yourself and extract it into this structure:

```text
data/
  Positive/  # crack = 1
  Negative/  # no crack = 0
```

`data/` is excluded from Git.

## Set up and run

Use Python 3.11 for the closest match to our experiment. PyTorch 2.5.1 (CPU) and torchvision 0.20.1 were used for the published numbers. Use the appropriate PyTorch installation from [pytorch.org](https://pytorch.org/get-started/locally/) for your own machine; other versions or GPUs can cause small differences.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
```

The full experiment is:

```bash
python -m crack_detection.train \
  --dataset-root data --model all --seed 42 \
  --epochs 5 --batch-size 64 --learning-rate 0.001 \
  --workers 2 --verify-images --output-dir runs/final_seed42
```

This takes a while on CPU: our CNN took about 59 minutes to fit, and MobileNetV2 about 87 minutes. You can run the models separately using `--model logistic`, `--model cnn`, or `--model mobilenet`. The [full-dataset workflow](.github/workflows/full_dataset.yml) shows the exact GitHub Actions commands we used.

MobileNetV2 downloads ImageNet-1K V1 weights on its first run. **Do not use `--no-pretrained` when trying to reproduce the transfer-learning result.** That changes the experiment.

After training, create the figures and report from the saved metrics:

```bash
python scripts/make_figures.py \
  --metrics runs/final_seed42/metrics.json --output-dir report/figures
python scripts/export_report_results.py \
  --metrics runs/final_seed42/metrics.json --output report/results_generated.tex
cd report && pdflatex final_report.tex && pdflatex final_report.tex
```

## Saved files

- `report/final_report.pdf` — the IEEE paper (five pages of main content; references after that).
- `report/final_report.tex` — editable LaTeX source.
- `results/full40k_seed42_metrics.json` — all three models, settings, timing, and test counts.
- `results/full40k_seed42_splits.csv` — the actual image paths used for training, validation, and testing.
- `results/*_errors.csv` — wrong predictions on the 6,000-image test set.
- `results/dataset_provenance.json` — the dataset mirror and archive hash.
- `report/figures/` — plots made from measured data and examples of real errors.

We also ran a small controlled CNN experiment with and without random horizontal/vertical flips. It uses just 1,000 images per class, the same seed and split in both conditions, and is **not** a second full-dataset result. Its JSON output is `results/ablation_subset_2k_seed42.json`. The [pilot workflow](.github/workflows/pilot_ablation.yml) has the exact commands.

## How the models were compared

The full 40,000-image run used 28,000 training patches, 6,000 validation patches, and 6,000 test patches, stratified by label. For the CNN and MobileNetV2, we kept the epoch with the best validation F1 and then tested once. The logistic regression model was fitted on the training set, using training-set statistics only for standardization. Neural-network training used Adam, cross-entropy, batch size 64, learning rate 0.001, and five epochs.

The scripts save metrics and the list of incorrect test predictions. They also save the seed, software versions, and model settings, so the reported numbers can be checked against a rerun. No image or pretrained weights are needed to read the results and paper; you only need them to train again.

## References

- Özgenel, Ç. F. (2018). *Concrete Crack Images for Classification*, Mendeley Data V1. [DOI](https://doi.org/10.17632/5y9wdsg2zt.1)
- Özgenel, Ç. F., and Gönenç Sorguç, A. (2018). *Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings.* [DOI](https://doi.org/10.22260/ISARC2018/0094)
- Zhang, L. et al. (2016). *Road Crack Detection Using Deep Convolutional Neural Network.* [DOI](https://doi.org/10.1109/ICIP.2016.7533052)
- Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* [DOI](https://doi.org/10.1109/CVPR.2018.00474)