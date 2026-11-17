"""Render the small, matched CNN pilot without mixing it with full-data scores."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    x = json.loads(a.input.read_text())
    if x.get("full_dataset_comparison") is not False or x["subset_total"] != 2000:
        raise ValueError("Expected a small pilot, not the 40k comparison")
    if x["seed"] != 42 or x["epochs"] != 5 or x["split_counts"]["test"]["n"] != 300:
        raise ValueError("Pilot settings do not match the paper")
    f = x["conditions"]["with_flips"]["test"]
    n = x["conditions"]["without_flips"]["test"]
    difference = f["f1"] - n["f1"]
    body = r"""% Generated from the measured CNN pilot, not the 40k experiment.
\renewcommand{\AblationTable}{%
\begin{table}[!t]
\caption{Matched CNN pilot on 2,000 patches (300 test patches)}
\label{tab:ablation}
\centering
\footnotesize
\begin{tabular}{lrrr}
\toprule
Training setting & Test accuracy & Recall & F1 \\
\midrule
"""
    body += f"Flips & {f['accuracy']:.4f} & {f['recall']:.4f} & {f['f1']:.4f} \\\n"
    body += f"No flips & {n['accuracy']:.4f} & {n['recall']:.4f} & {n['f1']:.4f} \\\n"
    body += r"""\bottomrule
\end{tabular}
\end{table}}
"""
    body += (r"\renewcommand{\AblationSummary}{"
             + f"Test F1 differed by {difference:+.4f} (flips minus no flips). "
             + "This is a single-seed subset check, not a measure of performance on independent source photographs."
             + "}\n")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(body)
    print(body)

if __name__ == "__main__":
    main()
