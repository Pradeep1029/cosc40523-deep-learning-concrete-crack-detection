"""Compare the two CNN settings on exactly the same held-out subset."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flips", type=Path, required=True)
    p.add_argument("--no-flips", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    paths = [a.flips, a.no_flips]
    x, y = [json.loads((d / "metrics.json").read_text()) for d in paths]
    if x["dataset_total_used"] != y["dataset_total_used"] or x["dataset_total_used"] >= 40000:
        raise ValueError("These must be matching subset runs")
    if x["seed"] != y["seed"] or x["seed"] != 42:
        raise ValueError("Seed mismatch")
    checksums = [sha(d / "splits.csv") for d in paths]
    if checksums[0] != checksums[1]:
        raise ValueError("The runs used different split assignments")
    m, n = x["models"][0], y["models"][0]
    if m["model"] != "cnn" or n["model"] != "cnn":
        raise ValueError("Expected CNN runs")
    if not m["configuration"]["augment_flips"] or n["configuration"]["augment_flips"]:
        raise ValueError("The two conditions are not flips vs no flips")
    for key in ("epochs", "batch_size", "lr", "seed", "image_size"):
        if m["configuration"][key] != n["configuration"][key]:
            raise ValueError("Other hyperparameters changed: " + key)
    answer = {
        "experiment": "CNN flip augmentation comparison on a 2,000-patch pilot subset",
        "full_dataset_comparison": False,
        "dataset_total_discovered": x["dataset_total_discovered"],
        "subset_total": x["dataset_total_used"],
        "split_counts": x["split_counts"],
        "seed": 42, "split_sha256": checksums[0],
        "epochs": m["configuration"]["epochs"],
        "conditions": {
            "with_flips": {
                "test": m["test_metrics"], "val": m["val_metrics"],
                "best_epoch": m["best_epoch"], "fit_seconds": m["fit_seconds"]},
            "without_flips": {
                "test": n["test_metrics"], "val": n["val_metrics"],
                "best_epoch": n["best_epoch"], "fit_seconds": n["fit_seconds"]},
        },
        "test_f1_difference_flips_minus_no_flips":
            m["test_metrics"]["f1"] - n["test_metrics"]["f1"],
        "test_accuracy_difference_flips_minus_no_flips":
            m["test_metrics"]["accuracy"] - n["test_metrics"]["accuracy"],
        "limitations": "Exploratory single-seed pilot on held-out patches, not independent source photos"
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(answer, indent=2) + "\n")
    print(json.dumps(answer, indent=2), flush=True)

if __name__ == "__main__":
    main()
