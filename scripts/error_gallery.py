"""Show a fixed random sample of actual mistakes from the full-data test set."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--errors-dir", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    a = p.parse_args()
    a.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(42)
    chosen = []
    fig, axes = plt.subplots(3, 4, figsize=(8.4, 6.0))
    for i, model in enumerate(("logistic", "cnn", "mobilenet")):
        with (a.errors_dir / f"{model}_errors.csv").open(newline="") as f:
            errors = list(csv.DictReader(f))
        for j, kind in enumerate(("false_positive", "false_negative")):
            options = sorted(
                (r for r in errors if r["error_type"] == kind),
                key=lambda r: r["relative_path"])
            choices = generator.choice(
                len(options), size=min(2, len(options)), replace=False) if options else []
            for k in range(2):
                ax = axes[i, j * 2 + k]
                ax.axis("off")
                if k < len(choices):
                    row = options[int(choices[k])]
                    path = a.dataset_root / row["relative_path"]
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    with Image.open(path) as im:
                        ax.imshow(im.convert("RGB"))
                    ax.set_title(("FP" if j == 0 else "FN") + f" {k+1}", fontsize=10)
                    chosen.append({"model": model, **row})
                else:
                    ax.text(.5, .5, "None in test set", transform=ax.transAxes,
                            ha="center", va="center", fontsize=8)
        axes[i, 0].text(-.14, .5, model.title(), transform=axes[i, 0].transAxes,
                        rotation=90, ha="right", va="center", fontsize=11)
    fig.suptitle("Held-out mistakes (FP: false positive; FN: false negative)", fontsize=12)
    fig.subplots_adjust(wspace=.08, hspace=.32, left=.11, right=.98, bottom=.04, top=.9)
    for ext in ("pdf", "png"):
        fig.savefig(a.output_prefix.with_suffix("." + ext), dpi=180)
    plt.close(fig)
    a.output_prefix.with_suffix(".json").write_text(
        json.dumps({"selection_seed": 42, "examples": chosen}, indent=2) + "\n")
    print(json.dumps(chosen, indent=2), flush=True)

if __name__ == "__main__":
    main()
