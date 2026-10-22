"""Reproducible train/validation/test comparisons for concrete crack images.

Run with: python -m crack_detection.train --dataset-root data --model all
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import warnings
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from .data import ConcreteDataset, describe_splits, export_split_csv, find_samples, split_samples
from .models import make_model


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}


def logistic_features(samples, feature_size: int = 16):
    x, y = [], []
    for sample in samples:
        with Image.open(sample.path) as im:
            pixels = np.asarray(im.convert("L").resize((feature_size, feature_size)), dtype=np.float32) / 255.0
        x.append(pixels.ravel()); y.append(sample.label)
    return np.stack(x), np.asarray(y, dtype=np.int64)


def run_logistic(parts, out_dir: Path):
    x_train, y_train = logistic_features(parts["train"])
    x_val, y_val = logistic_features(parts["val"])
    x_test, y_test = logistic_features(parts["test"])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    # Use validation to report a consistent baseline; do not tune on the test set.
    test_pred = model.predict(x_test)
    save_errors(parts["test"], y_test, test_pred, out_dir/"logistic_errors.csv")
    return {"model": "logistic", "val_metrics": metrics(y_val, val_pred),
            "test_metrics": metrics(y_test, test_pred), "notes": "16x16 grayscale pixel baseline"}


def transforms_for(model_name):
    size = 224 if model_name == "mobilenet" else 128
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if model_name == "mobilenet" else transforms.Normalize([0.5]*3, [0.5]*3)
    train = transforms.Compose([transforms.Resize((size, size)),
                                transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                                transforms.ToTensor(), norm])
    test = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm])
    return train, test


def evaluate(model, loader, device):
    model.eval(); yt, yp = [], []
    with torch.inference_mode():
        for x, y in loader:
            predictions = model(x.to(device)).argmax(dim=1).cpu().tolist()
            yp.extend(predictions); yt.extend(y.tolist())
    return metrics(yt, yp), np.asarray(yt), np.asarray(yp)


def run_neural(parts, model_name, out_dir, seed=42, epochs=5, batch_size=64,
               lr=1e-3, workers=0, pretrained=True):
    train_transform, test_transform = transforms_for(model_name)
    train_ds = ConcreteDataset(parts["train"], train_transform)
    val_ds = ConcreteDataset(parts["val"], test_transform)
    test_ds = ConcreteDataset(parts["test"], test_transform)
    g = torch.Generator().manual_seed(seed)
    loader_args = dict(batch_size=batch_size, num_workers=workers,
                       pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, generator=g, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(model_name, pretrained=pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_f1, best_state, best_epoch, history = -1.0, None, -1, []
    for epoch in range(1, epochs+1):
        model.train(); total_loss, total_count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            total_loss += float(loss.item()) * x.size(0); total_count += x.size(0)
        validation, _, _ = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": total_loss/total_count,
                        "val_accuracy": validation["accuracy"], "val_f1": validation["f1"]})
        print(f"{model_name}: epoch {epoch}/{epochs}, val_f1={validation['f1']:.3f}", flush=True)
        if validation["f1"] > best_f1:
            best_f1, best_epoch = validation["f1"], epoch
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    validation, _, _ = evaluate(model, val_loader, device)
    test, y_true, y_pred = evaluate(model, test_loader, device)
    save_errors(parts["test"], y_true, y_pred, out_dir/f"{model_name}_errors.csv")
    # Intentionally do not serialize weights/checkpoints in coursework delivery.
    return {"model": model_name, "best_epoch": best_epoch, "val_metrics": validation,
            "test_metrics": test, "history": history,
            "pretrained_imagenet_weights": bool(pretrained) if model_name == "mobilenet" else False,
            "device": str(device)}


def save_errors(samples, y_true, y_pred, filename):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["image", "true_label", "predicted_label"])
        for sample, truth, pred in zip(samples, y_true, y_pred):
            if int(truth) != int(pred):
                writer.writerow([str(sample.path), int(truth), int(pred)])


def main(argv=None):
    p = argparse.ArgumentParser(description="COSC 40523 Deep Learning: concrete crack classification")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--model", choices=["logistic", "cnn", "mobilenet", "all"], default="all")
    p.add_argument("--source-groups", type=Path, default=None,
                   help="CSV mapping original source photos; strongly recommended for leakage control")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    p.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet MobileNetV2 weights")
    args = p.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        p.error("epochs and batch-size must be positive")
    seed_all(args.seed)
    if args.source_groups is None:
        warnings.warn("Using image-level random splitting: patches from the same original photographs may cross splits. Results cannot be claimed source-independent. Provide --source-groups when original-photo IDs are available.", stacklevel=1)
    samples = find_samples(args.dataset_root, args.source_groups)
    parts = split_samples(samples, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_split_csv(parts, args.dataset_root, args.output_dir/"splits.csv")
    experiment = {"dataset": str(args.dataset_root.resolve()), "seed": args.seed,
                  "split_type": "source_group" if args.source_groups else "image_level",
                  "split_counts": describe_splits(parts), "models": []}
    requested = ["logistic", "cnn", "mobilenet"] if args.model == "all" else [args.model]
    for name in requested:
        if name == "logistic":
            result = run_logistic(parts, args.output_dir)
        else:
            result = run_neural(parts, name, args.output_dir, args.seed, args.epochs,
                                args.batch_size, args.learning_rate, args.workers,
                                not args.no_pretrained)
        experiment["models"].append(result)
        with (args.output_dir/"metrics.json").open("w", encoding="utf-8") as f:
            json.dump(experiment, f, indent=2)
    print(json.dumps({"split_counts": experiment["split_counts"], "models": [
        {"name": m["model"], "test": m["test_metrics"]} for m in experiment["models"]]}, indent=2))
    return experiment


if __name__ == "__main__":
    main()
