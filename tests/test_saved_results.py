"""Check that submitted numerical results agree with the saved experiment records."""
import csv
import hashlib
import json
from pathlib import Path
from pytest import approx

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))

def test_measured_results_match_error_counts():
    data = load("full40k_seed42_metrics.json")
    assert data["status"] == "complete"
    assert data["dataset_total_used"] == 40000
    assert data["seed"] == 42
    assert {x["model"] for x in data["models"]} == {"logistic", "cnn", "mobilenet"}
    assert [data["split_counts"][n]["n"] for n in ("train", "val", "test")] == [28000, 6000, 6000]
    for model in data["models"]:
        tn, fp = model["test_metrics"]["confusion_matrix"][0]
        fn, tp = model["test_metrics"]["confusion_matrix"][1]
        x = model["test_metrics"]
        assert tn + fp + fn + tp == 6000
        assert x["accuracy"] == approx((tn + tp) / 6000)
        assert x["precision"] == approx(tp / (tp + fp))
        assert x["recall"] == approx(tp / (tp + fn))
        assert x["f1"] == approx(2 * tp / (2 * tp + fp + fn))

def test_final_split_has_all_images_once():
    with (RESULTS / "full40k_seed42_splits.csv").open(newline="") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 40000
    assert len({x["relative_path"] for x in records}) == 40000
    for split, n in (("train", 28000), ("val", 6000), ("test", 6000)):
        selected = [x for x in records if x["split"] == split]
        assert len(selected) == n
        assert sum(int(x["label"]) for x in selected) == n // 2

def test_small_pilot_is_separate_and_identically_split():
    pilot = load("ablation_subset_2k_seed42.json")
    assert pilot["full_dataset_comparison"] is False
    assert pilot["subset_total"] == 2000
    assert pilot["seed"] == 42
    assert pilot["split_counts"]["test"]["n"] == 300
    digest = hashlib.sha256((RESULTS / "pilot_seed42_splits.csv").read_bytes()).hexdigest()
    assert digest == pilot["split_sha256"]
    f1 = pilot["conditions"]["with_flips"]["test"]["f1"]
    no_f1 = pilot["conditions"]["without_flips"]["test"]["f1"]
    assert pilot["test_f1_difference_flips_minus_no_flips"] == approx(f1 - no_f1)
