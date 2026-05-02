"""Render a single PNG that summarises a TIPSv2-on-Mangrove bench run.

Output (one figure with two panels):
  - Left:  confusion matrix heatmap (annotated with counts).
  - Right: world scatter — every labelled point coloured by its ground-truth
           class and shaped by correctness (circle = correct, ✗ = wrong),
           on a simple lon/lat axis with a faint coastline-style grid.

Usage::

    python scripts/viz_mangrove_bench.py \\
        --input data/olmoearth_projects/mangrove/bench_tipsv2_b14_n200.json \\
        --output data/olmoearth_projects/mangrove/bench_tipsv2_b14_n200.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Class colors match the prompt set's metadata so the chart agrees with
# whatever the TIPSv2 panel already shows in Roger Studio.
CLASS_COLORS = {
    "Mangrove": "#94eb63",
    "Water":    "#63d8eb",
    "Other":    "#eba963",
}


def _plot_confusion(ax, confusion: np.ndarray, classes: list[str], title: str) -> None:
    cm = np.asarray(confusion, dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = int(cm[i, j])
            frac = cm_norm[i, j]
            color = "white" if frac > 0.5 else "black"
            ax.text(j, i, f"{count}\n({frac*100:.0f}%)",
                    ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised fraction")


def _plot_world(ax, samples: list[dict]) -> None:
    # Faint background gridlines as a stand-in for coastlines (cartopy not
    # required). Latitude ±60 keeps the mangrove latitude band centered.
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 60)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(True, color="#cbd5e1", linewidth=0.5, alpha=0.6)
    ax.set_facecolor("#f8fafc")
    ax.axhline(0, color="#94a3b8", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="#94a3b8", linewidth=0.7, alpha=0.5)

    # Plot correct + wrong on top of each other so wrong markers stand out.
    correct_x: list[float] = []
    correct_y: list[float] = []
    correct_c: list[str] = []
    wrong_x: list[float] = []
    wrong_y: list[float] = []
    wrong_c: list[str] = []
    for s in samples:
        ref = s.get("ref")
        pred = s.get("pred_global")
        if ref is None or pred is None:
            continue
        c = CLASS_COLORS.get(ref, "#888888")
        if ref == pred:
            correct_x.append(s["lon"]); correct_y.append(s["lat"]); correct_c.append(c)
        else:
            wrong_x.append(s["lon"]); wrong_y.append(s["lat"]); wrong_c.append(c)

    ax.scatter(correct_x, correct_y, c=correct_c, marker="o", s=24,
               edgecolor="#22c55e", linewidth=0.8, alpha=0.95, label=f"correct ({len(correct_x)})")
    ax.scatter(wrong_x, wrong_y, c=wrong_c, marker="x", s=40,
               linewidth=1.5, alpha=0.95, label=f"wrong ({len(wrong_x)})")

    # Class colour legend (separate from correctness marker shape).
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                   markeredgecolor="#475569", markersize=8, label=name)
        for name, col in CLASS_COLORS.items()
    ]
    handles += [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#cbd5e1",
                   markeredgecolor="#22c55e", markersize=8, linewidth=2, label="correct"),
        plt.Line2D([0], [0], marker="x", color="#475569", markersize=10,
                   linewidth=2, label="wrong"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title("predictions per labelled point (color = ground truth, shape = correctness)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="bench result JSON")
    ap.add_argument("--output", required=True, help="PNG output path")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    classes = data["classes"]
    confusion = np.asarray(data["confusion"])
    samples = data["samples"]
    model = data["model"]
    acc = data["overall_accuracy"]
    f1 = data["macro_f1"]
    n = data["n_evaluated"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_confusion(
        axes[0], confusion, classes,
        f"{model.split('/')[-1]}  acc={acc*100:.1f}%  macro-F1={f1:.2f}  n={n}",
    )
    _plot_world(axes[1], samples)
    fig.suptitle(
        f"TIPSv2 zero-shot vs olmoearth_projects_mangrove ground truth",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out} (size={out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
