"""Make reproducible plots from a completed metrics.json, not imagined values."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(metrics_file: Path, out_dir: Path):
    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    if data.get("status") != "complete" or not data.get("models"):
        raise ValueError("Run experiments first; no complete metrics available")
    out_dir.mkdir(parents=True, exist_ok=True)
    models = data["models"]
    names = [m["model"] for m in models]
    vals = [m["test_metrics"]["f1"] for m in models]
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.bar(names, vals, color="0.40")
    ax.set(ylim=(0, 1.05), ylabel="Held-out test F1", title="Comparison on a shared test set")
    for i, v in enumerate(vals):
        ax.text(i, v + .02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir/"test_f1.pdf"); plt.close(fig)
    for m in models:
        cm = np.asarray(m["test_metrics"]["confusion_matrix"], dtype=int)
        fig, ax = plt.subplots(figsize=(3.7, 3.3))
        ax.imshow(cm, cmap="Greys")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
               xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
               title=f"{m['model']} test confusion matrix")
        for (i, j), x in np.ndenumerate(cm):
            ax.text(j, i, str(x), ha="center", va="center", color="white" if x > cm.max()/2 else "black")
        fig.tight_layout(); fig.savefig(out_dir/f"{m['model']}_confusion.pdf"); plt.close(fig)
        if m.get("history"):
            h = m["history"]
            fig, ax = plt.subplots(figsize=(5.3, 3.0))
            ax.plot([r["epoch"] for r in h], [r["val_f1"] for r in h], "o-", label="Val. F1")
            ax.plot([r["epoch"] for r in h], [r["val_accuracy"] for r in h], "s--", label="Val. accuracy")
            ax.set(xlabel="Epoch", ylabel="Metric", ylim=(0, 1.05), title=f"{m['model']} validation learning")
            ax.legend(); fig.tight_layout()
            fig.savefig(out_dir/f"{m['model']}_learning.pdf"); plt.close(fig)
    table = ["Model & Accuracy & Precision & Recall & F1 & Train s \\\\ ", "\\hline"]
    for m in models:
        x = m["test_metrics"]
        table.append("{} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.1f} \\\\".format(m["model"],
                       x["accuracy"], x["precision"], x["recall"], x["f1"], m["fit_seconds"]))
    (out_dir/"results_table.tex").write_text("\n".join(table)+"\n")
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("report/figures"))
    a = p.parse_args()
    render(a.metrics, a.output_dir)
