#!/usr/bin/env python3
"""
plot_results.py -- turns results/raw_outputs/results.csv (produced by
run_experiments.sh) into the figures and timing table used in docs/paper.tex.

This script is NOT required to reproduce the core experiment (that is done
entirely in C by run_experiments.sh); it only formats already-measured,
real data for the paper. Dependencies: numpy, matplotlib (see README.md).

Usage:
    python3 scripts/plot_results.py
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "results", "raw_outputs", "results.csv")
FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

rows = []
with open(CSV_PATH, newline="") as f:
    for r in csv.DictReader(f):
        r["workers"] = int(r["workers"])
        r["width"] = int(r["width"])
        r["height"] = int(r["height"])
        r["seconds"] = float(r["seconds"])
        rows.append(r)


def get(experiment, model, filt, workers=None, width=None, height=None):
    out = [r for r in rows if r["experiment"] == experiment and r["model"] == model and r["filter"] == filt]
    if workers is not None:
        out = [r for r in out if r["workers"] == workers]
    if width is not None:
        out = [r for r in out if r["width"] == width]
    if height is not None:
        out = [r for r in out if r["height"] == height]
    return out


FILTERS = ["gaussian_blur", "sobel", "sharpen", "grayscale"]
MODELS = [("pthreads", "Pthreads", "tab:blue", "o"),
          ("openmp", "OpenMP", "tab:orange", "s"),
          ("mpi", "MPI", "tab:green", "^")]
PRIMARY_FILTER = "gaussian_blur"
PRIMARY_LABEL = "Gaussian Blur (5x5)"
WORKERS = [1, 2, 4, 8]

# ---------------------------------------------------------------------
# Figure 1: Strong scaling -- speedup vs workers at fixed 3840x2160
# ---------------------------------------------------------------------
t_serial = get("strong_scaling", "sequential", PRIMARY_FILTER, width=3840, height=2160)[0]["seconds"]

plt.figure(figsize=(5.2, 4.0))
for key, label, color, marker in MODELS:
    speedups = []
    for p in WORKERS:
        t = get("strong_scaling", key, PRIMARY_FILTER, workers=p, width=3840, height=2160)[0]["seconds"]
        speedups.append(t_serial / t)
    plt.plot(WORKERS, speedups, marker=marker, color=color, label=label, linewidth=1.8)
plt.plot(WORKERS, WORKERS, linestyle="--", color="gray", label="Ideal linear")
plt.xlabel("Workers (threads / MPI processes)")
plt.ylabel("Speedup  S(p) = T$_{serial}$ / T$_{parallel}$")
plt.title(f"Strong Scaling -- {PRIMARY_LABEL}, 3840x2160")
plt.xticks(WORKERS)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "strong_scaling_speedup.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------
# Figure 2: Weak scaling -- execution time vs workers (rows/worker fixed)
# ---------------------------------------------------------------------
WEAK_DIMS = {1: (1920, 270), 2: (1920, 540), 4: (1920, 1080), 8: (1920, 2160)}

plt.figure(figsize=(5.2, 4.0))
for key, label, color, marker in MODELS:
    times = []
    for p in WORKERS:
        w, h = WEAK_DIMS[p]
        t = get("weak_scaling", key, PRIMARY_FILTER, workers=p, width=w, height=h)[0]["seconds"]
        times.append(t)
    plt.plot(WORKERS, times, marker=marker, color=color, label=label, linewidth=1.8)
plt.xlabel("Workers (threads / MPI processes)")
plt.ylabel("Execution time (s)  --  lower & flatter is better")
plt.title(f"Weak Scaling -- {PRIMARY_LABEL}, 270 rows/worker")
plt.xticks(WORKERS)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "weak_scaling_time.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------
# Figure 3: Problem-size sweep -- execution time vs N at workers=8
# ---------------------------------------------------------------------
size_tags = sorted({(r["width"], r["height"]) for r in rows if r["experiment"] == "size_sweep"},
                    key=lambda wh: wh[0] * wh[1])
ns = [w * h for w, h in size_tags]

plt.figure(figsize=(5.2, 4.0))
seq_times = [get("size_sweep", "sequential", PRIMARY_FILTER, width=w, height=h)[0]["seconds"] for w, h in size_tags]
plt.plot(ns, seq_times, marker="x", color="black", label="Sequential", linewidth=1.8)
for key, label, color, marker in MODELS:
    times = [get("size_sweep", key, PRIMARY_FILTER, workers=8, width=w, height=h)[0]["seconds"] for w, h in size_tags]
    plt.plot(ns, times, marker=marker, color=color, label=f"{label} (8 workers)", linewidth=1.8)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Image size N (pixels, log scale)")
plt.ylabel("Execution time (s, log scale)")
plt.title(f"Effect of Problem Size -- {PRIMARY_LABEL}")
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "size_sweep.png"), dpi=200)
plt.close()

# ---------------------------------------------------------------------
# Figure 4 (optional): real-dataset throughput -- images/sec vs workers,
# over the >=5000-real-photograph Imagenette-derived batch (see
# data/prepare_real_dataset.py). Skipped gracefully if that optional
# experiment wasn't run (no internet access / dataset not prepared).
# ---------------------------------------------------------------------
BATCH_FILTERS = ["gaussian_blur_batch", "sobel_batch", "sharpen_batch", "grayscale_batch"]
PRIMARY_BATCH_FILTER = "gaussian_blur_batch"
have_real_dataset = any(r["experiment"] == "dataset_throughput" for r in rows)

if have_real_dataset:
    seq_row = get("dataset_throughput", "sequential", PRIMARY_BATCH_FILTER)[0]
    dataset_count = seq_row["height"]
    seq_throughput = dataset_count / seq_row["seconds"]

    plt.figure(figsize=(5.2, 4.0))
    plt.axhline(seq_throughput, linestyle="--", color="black", label="Sequential", linewidth=1.8)
    for key, label, color, marker in MODELS:
        throughputs = []
        for p in WORKERS:
            r = get("dataset_throughput", key, PRIMARY_BATCH_FILTER, workers=p)[0]
            throughputs.append(r["height"] / r["seconds"])
        plt.plot(WORKERS, throughputs, marker=marker, color=color, label=label, linewidth=1.8)
    plt.xlabel("Workers (threads / MPI processes)")
    plt.ylabel("Throughput (images/sec)")
    plt.title(f"Real-Dataset Throughput -- {PRIMARY_LABEL}, {dataset_count} real photographs")
    plt.xticks(WORKERS)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "real_dataset_throughput.png"), dpi=200)
    plt.close()

    # LaTeX table: throughput (images/sec) for all four filters, all configs
    lines = []
    lines.append(r"\begin{tabular}{l l c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Model & Workers & Gauss. & Sobel & Sharpen & Gray. \\")
    lines.append(r"\midrule")

    seq_thr = {}
    for f in BATCH_FILTERS:
        r = get("dataset_throughput", "sequential", f)[0]
        seq_thr[f] = r["height"] / r["seconds"]
    lines.append(r"Sequential & 1 & %.1f & %.1f & %.1f & %.1f \\" % tuple(seq_thr[f] for f in BATCH_FILTERS))
    lines.append(r"\midrule")

    for key, label, _, _ in MODELS:
        for p in WORKERS:
            thr = {}
            for f in BATCH_FILTERS:
                r = get("dataset_throughput", key, f, workers=p)[0]
                thr[f] = r["height"] / r["seconds"]
            lines.append(r"%s & %d & %.1f & %.1f & %.1f & %.1f \\" % (label, p, *(thr[f] for f in BATCH_FILTERS)))
        lines.append(r"\midrule" if label != "MPI" else r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(os.path.join(FIG_DIR, "real_dataset_throughput_table.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    best_thr_speedup = max(
        (get("dataset_throughput", key, PRIMARY_BATCH_FILTER, workers=p)[0]["height"]
         / get("dataset_throughput", key, PRIMARY_BATCH_FILTER, workers=p)[0]["seconds"]) / seq_throughput
        for key, _, _, _ in MODELS for p in WORKERS
    )
    print(f"Real-dataset ({dataset_count} images) sequential throughput: {seq_throughput:.1f} img/s; "
          f"best parallel speedup over sequential: {best_thr_speedup:.2f}x")
else:
    print("Real-dataset throughput experiment not found in results.csv -- skipping Figure 4 / its table "
          "(this is the optional experiment described in data/prepare_real_dataset.py; the core "
          "strong/weak/size-sweep figures above do not depend on it).")

# ---------------------------------------------------------------------
# LaTeX table: execution time (s) for all four filters, all configs, 4K
# ---------------------------------------------------------------------
lines = []
lines.append(r"\begin{tabular}{l l c c c c}")
lines.append(r"\toprule")
lines.append(r"Model & Workers & Gauss. & Sobel & Sharpen & Gray. \\")
lines.append(r"\midrule")

seq_row = get("strong_scaling", "sequential", "gaussian_blur", width=3840, height=2160)
times = {f: get("strong_scaling", "sequential", f, width=3840, height=2160)[0]["seconds"] for f in FILTERS}
lines.append(r"Sequential & 1 & %.4f & %.4f & %.4f & %.4f \\" % tuple(times[f] for f in FILTERS))
lines.append(r"\midrule")

for key, label, _, _ in MODELS:
    for p in WORKERS:
        times = {f: get("strong_scaling", key, f, workers=p, width=3840, height=2160)[0]["seconds"] for f in FILTERS}
        lines.append(r"%s & %d & %.4f & %.4f & %.4f & %.4f \\" % (label, p, *(times[f] for f in FILTERS)))
    lines.append(r"\midrule" if label != "MPI" else r"\bottomrule")
lines.append(r"\end{tabular}")

with open(os.path.join(FIG_DIR, "exec_time_table.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

# ---------------------------------------------------------------------
# Summary printed to stdout (used to fact-check numbers quoted in the paper)
# ---------------------------------------------------------------------
# Headline number = best speedup within the primary, controlled strong-
# scaling experiment (fixed 3840x2160 image, all 4 filters, all 3 models,
# all worker counts). We deliberately do NOT hunt for the best speedup in
# the secondary problem-size sweep: several of its smaller sizes finish in
# well under a millisecond, where relative "speedup" is dominated by timer
# and scheduler noise rather than a reproducible parallelism effect (see
# Discussion). Restricting the headline claim to the controlled experiment
# avoids reporting a noise artifact as the paper's best result.
best_speedup = 0.0
best_desc = ""
for r in rows:
    if r["experiment"] != "strong_scaling" or r["model"] == "sequential":
        continue
    ts = get("strong_scaling", "sequential", r["filter"], width=r["width"], height=r["height"])
    if not ts:
        continue
    sp = ts[0]["seconds"] / r["seconds"]
    if sp > best_speedup:
        best_speedup = sp
        best_desc = f"{r['model']} {r['filter']} p={r['workers']} ({r['width']}x{r['height']})"

print(f"Best speedup observed: {best_speedup:.2f}x  [{best_desc}]")
print(f"Sequential T(4K, gaussian_blur) = {t_serial:.4f} s")
print("Figures written to:", FIG_DIR)
