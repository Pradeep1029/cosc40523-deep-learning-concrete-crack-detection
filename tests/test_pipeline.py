import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from crack_detection.data import find_samples, split_samples, describe_splits
from crack_detection.models import SmallCNN, make_model
from crack_detection.train import metrics, main, transforms_for


def make_images(root, num=18):
    for label, name in ((0, "Negative"), (1, "Positive")):
        (root/name).mkdir(parents=True, exist_ok=True)
        for i in range(num):
            rng = np.random.default_rng(12*i+label)
            arr = rng.integers(0, 50, (64, 64, 3), dtype=np.uint8)
            if label:
                arr[:, 28:32, :] = 220
            Image.fromarray(arr).save(root/name/f"{i:03d}.png")


def test_split_reproducible(tmp_path):
    make_images(tmp_path)
    samples = find_samples(tmp_path)
    a, b = split_samples(samples, seed=42), split_samples(samples, seed=42)
    assert [s.path for s in a["test"]] == [s.path for s in b["test"]]
    assert sum(s["n"] for s in describe_splits(a).values()) == 36
    assert all(len({s.label for s in part}) == 2 for part in a.values())


def test_group_split_disjoint_originals(tmp_path):
    make_images(tmp_path, num=30)
    mapping = tmp_path/"source_groups.csv"
    with mapping.open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["relative_path", "source_group"])
        for label in ("Positive", "Negative"):
            for i in range(30):
                writer.writerow([f"{label}/{i:03d}.png", f"{label}-source-{i//2}"])
    parts = split_samples(find_samples(tmp_path, mapping))
    groups = [{s.source_group for s in part} for part in parts.values()]
    assert groups[0].isdisjoint(groups[1]) and groups[0].isdisjoint(groups[2]) and groups[1].isdisjoint(groups[2])


def test_group_csv_rejects_duplicate_and_missing_rows(tmp_path):
    make_images(tmp_path)
    csv_file = tmp_path/"groups.csv"
    csv_file.write_text("relative_path,source_group\nPositive/000.png,source1\nPositive/000.png,source2\n")
    with pytest.raises(ValueError, match="Repeated"):
        find_samples(tmp_path, csv_file)


def test_cnn_forward():
    assert SmallCNN()(torch.zeros(2, 3, 64, 64)).shape == (2, 2)


def test_mobilenet_backbone_frozen_no_download():
    model = make_model("mobilenet", pretrained=False)
    assert not any(param.requires_grad for param in model.features.parameters())
    assert any(param.requires_grad for param in model.classifier.parameters())


def test_metrics_correct():
    x = metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert x["accuracy"] == pytest.approx(.75)
    assert x["confusion_matrix"] == [[1, 1], [0, 2]]


def test_transforms_dimensions():
    train, test = transforms_for("cnn", augment=False)
    image = Image.new("RGB", (227, 227))
    assert train(image).shape == (3, 128, 128)
    assert test(image).shape == (3, 128, 128)


def test_logistic_end_to_end_synthetic(tmp_path):
    root, out = tmp_path/"data", tmp_path/"out"
    make_images(root)
    result = main(["--dataset-root", str(root), "--model", "logistic", "--output-dir", str(out), "--verify-images"])
    assert (out/"metrics.json").exists()
    assert (out/"splits.csv").exists()
    assert result["models"][0]["model"] == "logistic"
    j = json.loads((out/"metrics.json").read_text())
    assert j["verification"]["checked"] == 36
    assert j["dataset_total_used"] == 36


def test_reject_tiny_subset(tmp_path):
    make_images(tmp_path)
    with pytest.raises(SystemExit):
        main(["--dataset-root", str(tmp_path), "--model", "logistic", "--limit-per-class", "5"])
