import csv
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from crack_detection.data import Sample, find_samples, split_samples, describe_splits
from crack_detection.models import SmallCNN
from crack_detection.train import metrics, main
import torch


def make_images(root, num=18):
    for label, name in ((0, "Negative"), (1, "Positive")):
        (root/name).mkdir(parents=True)
        for i in range(num):
            rng = np.random.default_rng(12*i + label)
            pixels = rng.integers(0, 50, (64, 64, 3), dtype=np.uint8)
            if label:
                pixels[:, 28:32, :] = 220
            Image.fromarray(pixels).save(root/name/f"{i:03d}.png")


def test_split_reproducible(tmp_path):
    make_images(tmp_path)
    samples = find_samples(tmp_path)
    p1 = split_samples(samples, seed=42)
    p2 = split_samples(samples, seed=42)
    assert [s.path for s in p1["test"]] == [s.path for s in p2["test"]]
    assert sum(x["n"] for x in describe_splits(p1).values()) == 36
    assert all(len({s.label for s in part}) == 2 for part in p1.values())


def test_group_split_disjoint_originals(tmp_path):
    make_images(tmp_path, num=30)
    manifest = tmp_path/"source_groups.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["relative_path", "source_group"])
        for label in ("Positive", "Negative"):
            for i in range(30):
                w.writerow([f"{label}/{i:03d}.png", f"{label}-original-{i//2}"])
    groups = find_samples(tmp_path, manifest)
    parts = split_samples(groups)
    sets = [{s.source_group for s in val} for val in parts.values()]
    assert sets[0].isdisjoint(sets[1]) and sets[0].isdisjoint(sets[2]) and sets[1].isdisjoint(sets[2])


def test_small_cnn_forward():
    model = SmallCNN()
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 2)


def test_metrics_correct():
    got = metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert got["accuracy"] == pytest.approx(0.75)
    assert got["confusion_matrix"] == [[1, 1], [0, 2]]


def test_logistic_end_to_end_on_synthetic_images(tmp_path):
    root = tmp_path/"data"; make_images(root)
    out = tmp_path/"out"
    result = main(["--dataset-root", str(root), "--model", "logistic", "--output-dir", str(out)])
    assert (out/"splits.csv").exists() and (out/"metrics.json").exists()
    assert len(result["models"]) == 1

