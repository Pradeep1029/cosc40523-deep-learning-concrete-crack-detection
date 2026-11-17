"""Image discovery, fixed train/validation/test manifests, and optional photo grouping."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    source_group: str | None = None


def find_samples(root: Path, group_csv: Path | None = None) -> list[Sample]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    groups: dict[str, str] = {}
    if group_csv:
        with Path(group_csv).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not {"relative_path", "source_group"}.issubset(reader.fieldnames or []):
                raise ValueError("Group CSV needs relative_path,source_group columns")
            for row in reader:
                key = Path(row["relative_path"]).as_posix()
                if key in groups:
                    raise ValueError(f"Repeated source-group mapping: {key}")
                if not row["source_group"].strip():
                    raise ValueError(f"Missing source_group for {key}")
                groups[key] = row["source_group"].strip()
    result = []
    for dirname, label in (("Negative", 0), ("Positive", 1)):
        class_dir = root / dirname
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected {class_dir}; required: Negative and Positive")
        for path in sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            rel = path.relative_to(root).as_posix()
            if group_csv and rel not in groups:
                raise ValueError(f"Group CSV has no source_group for {rel}")
            result.append(Sample(path, label, groups.get(rel)))
    if not result or len({s.label for s in result}) != 2:
        raise ValueError("Both Positive and Negative class folders must contain images")
    if group_csv and set(groups) != {s.path.relative_to(root).as_posix() for s in result}:
        raise ValueError("Group mapping contains paths not found in dataset")
    return result


def split_samples(samples: Sequence[Sample], seed: int = 42,
                  val_fraction: float = .15, test_fraction: float = .15) -> dict[str, list[Sample]]:
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Invalid split fractions")
    if len(samples) < 12:
        raise ValueError("At least 12 images required")
    has_groups = [s.source_group is not None for s in samples]
    if any(has_groups) and not all(has_groups):
        raise ValueError("All samples must have source groups or none may have them")
    indices = np.arange(len(samples))
    labels = np.asarray([s.label for s in samples])
    groups = np.asarray([s.source_group for s in samples])
    if all(has_groups):
        a, test = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                         random_state=seed).split(indices, labels, groups))
        b, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=val_fraction/(1-test_fraction),
                                             random_state=seed+1).split(a, labels[a], groups[a]))
        train, val = a[b], a[val_idx]
    else:
        a, test = train_test_split(indices, test_size=test_fraction, stratify=labels, random_state=seed)
        train, val = train_test_split(a, test_size=val_fraction/(1-test_fraction),
                                      stratify=labels[a], random_state=seed+1)
    parts = {key: [samples[int(i)] for i in ids]
             for key, ids in (("train", train), ("val", val), ("test", test))}
    for key, portion in parts.items():
        if not portion or len({s.label for s in portion}) < 2:
            raise ValueError(f"{key} lacks both classes; check group map and split")
    if len({s.path for part in parts.values() for s in part}) != len(samples):
        raise ValueError("Samples overlap or were omitted in split")
    if all(has_groups):
        group_sets = [{s.source_group for s in part} for part in parts.values()]
        if not all(group_sets[i].isdisjoint(group_sets[j]) for i in range(3) for j in range(i+1, 3)):
            raise ValueError("Source photo groups overlap across splits")
    return parts


class ConcreteDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            x = self.transform(img.convert("RGB"))
        return x, sample.label


def describe_splits(parts: dict[str, list[Sample]]):
    return {key: {"n": len(items), "positive": sum(s.label for s in items),
                   "negative": len(items) - sum(s.label for s in items),
                   "source_groups": len({s.source_group for s in items}) if items[0].source_group else None}
            for key, items in parts.items()}


def export_split_csv(parts: dict[str, list[Sample]], dataset_root: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "relative_path", "label", "source_group"])
        for name, samples in parts.items():
            for sample in samples:
                writer.writerow([name, sample.path.relative_to(dataset_root.resolve()).as_posix(),
                                 sample.label, sample.source_group or ""])
