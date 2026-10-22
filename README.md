# COSC 40523 Deep Learning - Concrete Crack Detection

**TCU | Fall 2026 | Team Vectorise**  
**Members:** Jayapradeep Jayaraman Srinivas and Harshvardhan Datre

Our project looks at how well different models can tell whether an image of concrete contains a crack. We compare a simple logistic regression baseline, a CNN built from scratch, and a pretrained MobileNetV2 model.

The goal is to evaluate all three on the same test images, compare their results, and look at the mistakes each model makes.

## Models

| Model | Approach |
| --- | --- |
| Logistic regression | Uses grayscale images resized to 16 × 16 pixels as a baseline. Pixel values are standardized using the training set. |
| Custom CNN | Learns from RGB images using three convolution blocks, batch normalization, global average pooling, and a two-class output layer. |
| MobileNetV2 | Starts with ImageNet weights, freezes the feature extractor, and trains a new classification layer for crack detection. |

## Dataset

We use [Concrete Crack Images for Classification](https://doi.org/10.17632/5y9wdsg2zt.1) from Mendeley Data. It contains 40,000 RGB images, each 227 × 227 pixels: 20,000 with cracks and 20,000 without.

Download and extract the dataset on your computer. The folder passed to `--dataset-root` should contain the `Positive` and `Negative` folders directly:

```text
data/
  Positive/
    00001.jpg
    ...
  Negative/
    00001.jpg
    ...
```

`Positive` is the crack class (1), and `Negative` is the no-crack class (0). The dataset, model weights, and training outputs are kept out of the repository.

## Getting started

Use Python 3.10 or newer. From the project folder, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the PyTorch build for your computer using the instructions at [pytorch.org](https://pytorch.org/get-started/locally/). Then install the project and run the tests:

```bash
python -m pip install -e '.[test]'
pytest -q
```

## Running the models

To run all three models:

```bash
python -m crack_detection.train --dataset-root data --model all --epochs 5 --batch-size 64
```

You can also run them individually:

```bash
python -m crack_detection.train --dataset-root data --model logistic
python -m crack_detection.train --dataset-root data --model cnn --epochs 5
python -m crack_detection.train --dataset-root data --model mobilenet --epochs 5
```

MobileNetV2 downloads pretrained ImageNet weights by default. For a quick check without that download, use:

```bash
python -m crack_detection.train --dataset-root data --model mobilenet --no-pretrained --epochs 1
```

That last command starts from random weights, so it does not test transfer learning.

## How we evaluate the models

The default split is approximately 70% training, 15% validation, and 15% testing, with a random seed of 42. The image-level split keeps the class proportions similar across the three sets. Horizontal and vertical flips are applied only to training images for the neural models.

Validation F1 is used to select the best neural model checkpoint. The test set is then used to report accuracy, precision, recall, F1, and the confusion matrix. Each run also saves a list of misclassified test images so we can inspect false positives and false negatives.

### Keeping related images together

The dataset's 40,000 images were cropped from 458 original photographs. If crops from the same photo appear in both training and testing, the scores may look better than the model's performance on new photos.

When a reliable mapping to the original photos is available, the code can keep each photo's crops in a single split. Create a CSV covering every image with these columns:

```csv
relative_path,source_group
Positive/00001.jpg,original-photo-17
Negative/00001.jpg,original-photo-94
```

These rows only show the format; replace them with verified photo IDs. Run the models with:

```bash
python -m crack_detection.train --dataset-root data --source-groups original_photo_map.csv --model all
```

Without that mapping, the script uses the image-level split and prints a warning. Results from that split measure performance on held-out crops, which may still share a source photo with training images. Group-based splits keep photos separate, though class proportions may vary between sets.

## Results and progress

Each run saves its output in `runs/latest/`:

- `metrics.json`: model settings and evaluation scores.
- `splits.csv`: the images assigned to each split.
- `<model>_errors.csv`: misclassified test images for each model.

The training code and tests are included. Evaluation on the full concrete image dataset is still pending, so there are no verified dataset performance scores to report yet.

## References

1. Özgenel, Ç. F., & Gönenç Sorguç, A. (2018). Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings. *ISARC 2018*, 693–700. [Paper](https://doi.org/10.22260/ISARC2018/0094).
2. Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). Road Crack Detection Using Deep Convolutional Neural Network. *2016 IEEE International Conference on Image Processing*, 3708–3712. [Paper](https://doi.org/10.1109/ICIP.2016.7533052).
3. Özgenel, Ç. F. (2018). Concrete Crack Images for Classification (Version 1). *Mendeley Data*. [Dataset](https://doi.org/10.17632/5y9wdsg2zt.1).
