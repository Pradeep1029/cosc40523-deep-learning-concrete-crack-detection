"""Refuse to typeset full-dataset results unless provenance/configuration checks pass."""
import argparse
import csv
import json
from pathlib import Path

EXPECTED = {"logistic", "cnn", "mobilenet"}


def validate(path: Path, required_models=EXPECTED):
    x = json.loads(path.read_text(encoding="utf-8"))
    if x.get("status") != "complete":
        raise ValueError("Results are not complete")
    if x.get("dataset_total_discovered") != 40000 or x.get("dataset_total_used") != 40000:
        raise ValueError("Expected all 40,000 official image patches; do not label a subset full-dataset")
    if x.get("development_subset_per_class") is not None:
        raise ValueError("Development subset cannot be presented as full-dataset")
    if x.get("seed") != 42:
        raise ValueError("The report documents seed 42; refuse mismatched results")
    if set(m["model"] for m in x["models"]) != set(required_models):
        raise ValueError(f"Expected model set: {required_models}")
    if not all(m.get("test_metrics") for m in x["models"]):
        raise ValueError("Missing test evaluation")
    for m in x["models"]:
        if m["model"] == "mobilenet" and not m["configuration"].get("pretrained_imagenet_weights"):
            raise ValueError("MobileNetV2 lacks ImageNet pretrained weights")
    return x


def build(metrics_path: Path, output: Path):
    x = validate(metrics_path)
    models = {m["model"]: m for m in x["models"]}
    order = ["logistic", "cnn", "mobilenet"]
    rows = []
    for name in order:
        m = models[name]
        r = m["test_metrics"]
        display = {"logistic":"Logistic", "cnn":"Scratch CNN", "mobilenet":"MobileNetV2"}[name]
        rows.append(f"{display} & {r['accuracy']:.3f} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} \\\\")
    best = max(order, key=lambda name: models[name]["test_metrics"]["f1"])
    val = models[best]["test_metrics"]["f1"]
    parts = x["split_counts"]
    contents = r"""% Automatically generated from verified full-dataset results. Do not manually edit.
\renewcommand{\ResultsAvailable}{1}
\newcommand{\EmpiricalResultsTable}{%
\begin{table}[!t]
\caption{Full-Dataset Test Metrics (Crack Class Positive)}
\label{tab:results}
\centering
\footnotesize
\begin{tabular}{lrrrr}
\hline
Model & Acc. & Prec. & Recall & F1 \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}}
"""
    best_disp = {"logistic":"Logistic", "cnn":"Scratch CNN", "mobilenet":"MobileNetV2"}[best]
    contents += f"\\newcommand{{\\BestModel}}{{{best_disp}}}\n"
    contents += f"\\newcommand{{\\BestFOne}}{{{val:.3f}}}\n"
    contents += f"\\newcommand{{\\TrainingN}}{{{parts['train']['n']}}}\n"
    contents += f"\\newcommand{{\\ValidationN}}{{{parts['val']['n']}}}\n"
    contents += f"\\newcommand{{\\TestingN}}{{{parts['test']['n']}}}\n"
    contents += f"\\newcommand{{\\SplitProtocol}}{{{x['split'].replace('_','-')}}}\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(contents, encoding="utf-8")
    print(f"Validated 40,000-patch run and wrote {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("report/results_generated.tex"))
    a = p.parse_args()
    build(a.metrics, a.output)
