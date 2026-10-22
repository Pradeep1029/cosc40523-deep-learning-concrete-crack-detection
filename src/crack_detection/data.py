"""Dataset discovery and reproducible split manifests."""
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
    """Read Positive/Negative class folders (nested folders accepted).

    group_csv, when supplied, must have `relative_path,source_group` columns.
    It must identify the ORIGINAL source photograph for every patch; grouping
    by arbitrary filename or patch cannot prevent source-photo leakage.
    """
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
                if not row["source_group"].strip():
                    raise ValueError(f"Missing source_group for {key}")
                groups[key] = row["source_group"].strip()
    output: list[Sample] = []
    for dirname, label in (("Negative", 0), ("Positive", 1)):
        class_dir = root / dirname
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected {class_dir} (class labels: Negative, Positive)")
        paths = sorted(p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        for p in paths:
            rel = p.relative_to(root).as_posix()
            if group_csv and rel not in groups:
                raise ValueError(f"Group CSV has no source_group for {rel}")
            output.append(Sample(p, label, groups.get(rel)))
    if not output or len({s.label for s in output}) != 2:
        raise ValueError("Both Positive and Negative class folders need images")
    return output


def split_samples(samples: Sequence[Sample], seed: int = 42,
                  val_fraction: float = 0.15, test_fraction: float = 0.15
                  ) -> dict[str, list[Sample]]:
    """Return disjoint train/val/test. Group split iff all samples have source group.

    GroupShuffleSplit approximately matches target fractions; it does not
    enforce class balance, so the caller should inspect class counts.
    """
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Invalid fractions")
    if len(samples) < 12:
        raise ValueError("At least 12 images required for a useful train/val/test split")
    has_groups = [s.source_group is not None for s in samples]
    if any(has_groups) and not all(has_groups):
        raise ValueError("All samples must have source groups or none may have them")
    inds = np.arange(len(samples))
    labels = np.array([s.label for s in samples]); groups = np.array([s.source_group for s in samples])
    if all(has_groups):
        a, test = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                         random_state=seed).split(inds, labels, groups))
        b, val_idx = next(GroupShuffleSplit(n_splits=1,
                                             test_size=val_fraction/(1-test_fraction),
                                             random_state=seed+1).split(a, labels[a], groups[a]))
        train = a[b]; val = a[val_idx]
    else:
        a, test = train_test_split(inds, test_size=test_fraction, stratify=labels, random_state=seed)
        train, val = train_test_split(a, test_size=val_fraction/(1-test_fraction),
                                      stratify=labels[a], random_state=seed+1)
    partitions = {key: [samples[int(i)] for i in indices]
                  for key, indices in (("train", train), ("val", val), ("test", test))}
    for key, part in partitions.items():
        if not part or len({s.label for s in part}) != 2:
            raise ValueError(f"Split {key} lacks an example of both classes; use more source groups")
    assert len({s.path for part in partitions.values() for s in part}) == len(samples)
    if all(has_groups):
        group_sets = [{s.source_group for s in part} for part in partitions.values()]
        assert all(group_sets[i].isdisjoint(group_sets[j]) for i in range(3) for j in range(i+1, 3))
    return partitions


class ConcreteDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with Image.open(sample.path) as im:
            x = self.transform(im.convert("RGB"))
        return x, sample.label


def describe_splits(parts: dict[str, list[Sample]]):
    return {split: {"n": len(xs), "positive": sum(s.label for s in xs),
                    "negative": len(xs)-sum(s.label for s in xs),
                    "source_groups": len({s.source_group for s in xs}) if xs[0].source_group else None}
            for split, xs in parts.items()}


def export_split_csv(parts: dict[str, list[Sample]], dataset_root: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["split", "relative_path", "label", "source_group"])
        for split, items in parts.items():
            for sample in items:
                writer.writerow([split, sample.path.relative_to(dataset_root.resolve()).as_posix(),
                                sample.label, sample.source_group or ""])
