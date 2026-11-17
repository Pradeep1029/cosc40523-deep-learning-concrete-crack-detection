"""Train and evaluate a reproducible baseline/CNN/MobileNetV2 comparison.

The held-out test set is evaluated only after training/validation selection.
Run: python -m crack_detection.train --dataset-root data --model all
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
import random
import time
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from .data import ConcreteDataset, describe_splits, export_split_csv, find_samples, split_samples
from .models import make_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def logistic_features(samples, feature_size=16):
    features, labels = [], []
    for sample in samples:
        with Image.open(sample.path) as image:
            gray = np.asarray(image.convert("L").resize((feature_size, feature_size)), dtype=np.float32) / 255.0
        features.append(gray.ravel())
        labels.append(sample.label)
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def save_errors(samples, y_true, y_pred, root, filename):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["relative_path", "true_label", "predicted_label", "error_type"])
        for sample, truth, pred in zip(samples, y_true, y_pred):
            if int(truth) != int(pred):
                writer.writerow([sample.path.relative_to(root.resolve()).as_posix(), int(truth), int(pred),
                                 "false_negative" if int(truth) == 1 else "false_positive"])


def run_logistic(parts, root, out_dir, seed=42):
    t0 = time.perf_counter()
    x_train, y_train = logistic_features(parts["train"])
    x_val, y_val = logistic_features(parts["val"])
    x_test, y_test = logistic_features(parts["test"])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - t0
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    save_errors(parts["test"], y_test, test_pred, root, out_dir / "logistic_errors.csv")
    return {
        "model": "logistic", "best_epoch": None,
        "val_metrics": metrics(y_val, val_pred), "test_metrics": metrics(y_test, test_pred),
        "fit_seconds": round(fit_seconds, 3), "history": [],
        "configuration": {"features": "grayscale 16x16", "normalization": "training-only StandardScaler",
                          "classifier": "LogisticRegression(max_iter=1000)", "seed": seed},
    }


def transforms_for(model_name, augment=True):
    size = 224 if model_name == "mobilenet" else 128
    normalizer = (transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
                  if model_name == "mobilenet" else transforms.Normalize([.5]*3, [.5]*3))
    common = [transforms.Resize((size, size))]
    training = common + ([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] if augment else [])
    training += [transforms.ToTensor(), normalizer]
    inference = common + [transforms.ToTensor(), normalizer]
    return transforms.Compose(training), transforms.Compose(inference)


def evaluate(model, loader, device):
    model.eval()
    true, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predicted.extend(logits.argmax(1).cpu().tolist())
            true.extend(labels.tolist())
    return metrics(true, predicted), np.asarray(true), np.asarray(predicted)


def run_neural(parts, model_name, root, out_dir, seed=42, epochs=5,
               batch_size=64, lr=1e-3, workers=0, pretrained=True, augment=True):
    train_transform, eval_transform = transforms_for(model_name, augment)
    train_ds = ConcreteDataset(parts["train"], train_transform)
    val_ds = ConcreteDataset(parts["val"], eval_transform)
    test_ds = ConcreteDataset(parts["test"], eval_transform)
    generator = torch.Generator().manual_seed(seed)
    loader_kw = {"batch_size": batch_size, "num_workers": workers,
                 "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(model_name, pretrained=pretrained).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history, best_state, best_epoch, best_f1 = [], None, None, -1.0
    train_start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        # A frozen transfer-learning backbone must keep batch-normalization
        # running statistics fixed. requires_grad=False alone is insufficient.
        if model_name == "mobilenet":
            model.features.eval()
        t0 = time.perf_counter()
        total_loss, n = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            n += len(images)
        validation, _, _ = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total_loss/n,
               "val_accuracy": validation["accuracy"], "val_f1": validation["f1"],
               "seconds": round(time.perf_counter()-t0, 3)}
        history.append(row)
        print(f"{model_name} epoch {epoch}/{epochs}: val_F1={row['val_f1']:.4f} ", flush=True)
        if validation["f1"] > best_f1:
            best_f1, best_epoch = validation["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No selected epoch")
    fit_seconds = time.perf_counter() - train_start
    model.load_state_dict(best_state)
    validation, _, _ = evaluate(model, val_loader, device)
    test, y_true, y_pred = evaluate(model, test_loader, device)
    save_errors(parts["test"], y_true, y_pred, root, out_dir / f"{model_name}_errors.csv")
    return {
        "model": model_name, "best_epoch": best_epoch,
        "val_metrics": validation, "test_metrics": test,
        "fit_seconds": round(fit_seconds, 3), "history": history,
        "configuration": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                          "optimizer": "Adam", "loss": "CrossEntropyLoss", "augment_flips": augment,
                          "image_size": 224 if model_name == "mobilenet" else 128,
                          "pretrained_imagenet_weights": bool(pretrained) if model_name == "mobilenet" else False,
                          "feature_extractor_frozen": model_name == "mobilenet",
                          "frozen_backbone_batchnorm_eval": model_name == "mobilenet",
                          "device": str(device), "seed": seed, "workers": workers},
    }


def verify_images(samples):
    unreadable = []
    dimensions = {}
    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                image.verify()
            with Image.open(sample.path) as image:
                dimensions[str(image.size)] = dimensions.get(str(image.size), 0)+1
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            unreadable.append((str(sample.path), str(exc)))
    if unreadable:
        raise ValueError(f"Unreadable images ({len(unreadable)}): {unreadable[:5]}")
    return {"checked": len(samples), "size_counts": dimensions, "unreadable": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Concrete crack image classification reproducibility experiment")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-groups", type=Path, default=None,
                        help="Verified CSV of original source-photo IDs; otherwise image-level split")
    parser.add_argument("--model", choices=["logistic", "cnn", "mobilenet", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/final_seed42"))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=None,
                        help="Development-only deterministic subset; must disclose in report")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("epochs, batch-size must be positive and workers non-negative")
    if args.limit_per_class is not None and args.limit_per_class < 10:
        parser.error("limit-per-class must be at least 10")
    seed_all(args.seed)
    if args.source_groups is None:
        warnings.warn("IMAGE-LEVEL SPLIT: crops from the same source photograph can cross partitions. "
                      "This is not a source-photograph-independent evaluation.", stacklevel=1)
    root = args.dataset_root.resolve()
    samples = find_samples(root, args.source_groups)
    original_count = len(samples)
    if args.limit_per_class is not None:
        warnings.warn("Development subset only: these results are NOT full-dataset results", stacklevel=1)
        rng = np.random.default_rng(args.seed)
        groups = []
        for label in (0, 1):
            class_samples = [s for s in samples if s.label == label]
            if len(class_samples) < args.limit_per_class:
                parser.error(f"Class {label} has fewer than {args.limit_per_class} images")
            selected = rng.choice(len(class_samples), args.limit_per_class, replace=False)
            groups.extend(class_samples[int(i)] for i in sorted(selected))
        samples = groups
    verification = verify_images(samples) if args.verify_images else None
    parts = split_samples(samples, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_split_csv(parts, root, args.output_dir/"splits.csv")
    experiment = {
        "status": "complete", "dataset_root": str(root),
        "dataset_reference": "Ozgenel 2018, doi:10.17632/5y9wdsg2zt.1",
        "dataset_total_discovered": original_count,
        "dataset_total_used": len(samples),
        "development_subset_per_class": args.limit_per_class,
        "seed": args.seed,
        "split": "source_photo_grouped" if args.source_groups else "image_level_stratified",
        "split_counts": describe_splits(parts),
        "verification": verification,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "sklearn": sklearn.__version__,
                        "cuda": bool(torch.cuda.is_available()), "platform": platform.platform()},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "models": []}
    requested = ["logistic", "cnn", "mobilenet"] if args.model == "all" else [args.model]
    for model_name in requested:
        if model_name == "logistic":
            result = run_logistic(parts, root, args.output_dir, seed=args.seed)
        else:
            result = run_neural(parts, model_name, root, args.output_dir, seed=args.seed,
                                epochs=args.epochs, batch_size=args.batch_size,
                                lr=args.learning_rate, workers=args.workers,
                                pretrained=not args.no_pretrained,
                                augment=not args.no_augmentation)
        experiment["models"].append(result)
        with (args.output_dir/"metrics.json").open("w", encoding="utf-8") as f:
            json.dump(experiment, f, indent=2, default=str)
    print(json.dumps({"split": experiment["split"], "counts": experiment["split_counts"],
                      "models": [{"model": m["model"], "test_metrics": m["test_metrics"]}
                                 for m in experiment["models"]]}, indent=2))
    return experiment


if __name__ == "__main__":
    main()
