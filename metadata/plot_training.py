#!/usr/bin/env python3
"""
Plot training metrics from metrics.csv (+ response-token-accuracy pulled
from train.log, since it isn't in the CSV) using a dark theme.

Usage:
    python plot_training.py --csv metrics.csv --log train.log --outdir plots
"""

import argparse
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# --------------------------------------------------------------------------
# Dark theme
# --------------------------------------------------------------------------
BG = "#0d1117"
PANEL = "#161b22"
GRID = "#30363d"
TEXT = "#c9d1d9"
ACCENT = {
    "train": "#58a6ff",   # blue
    "val": "#f78166",     # orange/red
    "lr": "#a5d6ff",      # light blue
    "gnorm": "#d2a8ff",   # purple
    "tokspeed": "#7ee787",# green
    "acc": "#ffa657",     # amber
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.titlecolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT,
    "font.size": 11,
    "savefig.facecolor": BG,
})


def style_axis(ax, title=None, xlabel="Step", ylabel=None):
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="major")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))


def smooth(series, window=9):
    """Simple rolling mean, ignoring NaNs, min_periods=1."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
def load_metrics_csv(path):
    df = pd.read_csv(path)
    # Two kinds of rows share a step: train rows (loss populated) and
    # eval rows (val_loss populated, other train-only cols are NaN).
    train_df = df.dropna(subset=["loss"]).reset_index(drop=True)
    val_df = df.dropna(subset=["val_loss"])[["step", "val_loss"]].reset_index(drop=True)
    return train_df, val_df


LOG_LINE_RE = re.compile(
    r"step\s+([\d,]+)\s*\|\s*loss\s+([\d.]+)\s*\|\s*lr\s+([\d.eE+-]+)\s*\|"
)
ACC_LINE_RE = re.compile(r"resp_tok_acc=\s*([\d.]+)%")


def load_resp_tok_acc(path):
    """Pull per-step response-token-accuracy from the raw train.log,
    since this metric never made it into metrics.csv."""
    steps, accs = [], []
    pending_step = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = LOG_LINE_RE.search(line)
            if m:
                pending_step = int(m.group(1).replace(",", ""))
                continue
            m2 = ACC_LINE_RE.search(line)
            if m2 and pending_step is not None:
                steps.append(pending_step)
                accs.append(float(m2.group(1)))
                pending_step = None
    return pd.DataFrame({"step": steps, "resp_tok_acc": accs})


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_loss(train_df, val_df, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_df["step"], train_df["loss"], color=ACCENT["train"],
            alpha=0.35, linewidth=1, label="_nolegend_")
    ax.plot(train_df["step"], smooth(train_df["loss"]), color=ACCENT["train"],
            linewidth=2, label="Train loss (smoothed)")
    ax.plot(val_df["step"], val_df["val_loss"], color=ACCENT["val"],
            marker="o", markersize=5, linewidth=2, label="Val loss")
    style_axis(ax, title="Loss vs Step", ylabel="Loss")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss.png"), dpi=150)
    plt.close(fig)


def plot_lr(train_df, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_df["step"], train_df["lr"], color=ACCENT["lr"], linewidth=2)
    style_axis(ax, title="Learning Rate Schedule", ylabel="Learning rate")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "learning_rate.png"), dpi=150)
    plt.close(fig)


def plot_grad_norm(train_df, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_df["step"], train_df["grad_norm"], color=ACCENT["gnorm"],
            alpha=0.4, linewidth=1)
    ax.plot(train_df["step"], smooth(train_df["grad_norm"]), color=ACCENT["gnorm"],
            linewidth=2, label="Grad norm (smoothed)")
    style_axis(ax, title="Gradient Norm vs Step", ylabel="Grad norm")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "grad_norm.png"), dpi=150)
    plt.close(fig)


def plot_throughput(train_df, outdir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_df["step"], train_df["tok_per_sec"] / 1000, color=ACCENT["tokspeed"],
            alpha=0.35, linewidth=1)
    ax.plot(train_df["step"], smooth(train_df["tok_per_sec"]) / 1000, color=ACCENT["tokspeed"],
            linewidth=2, label="Throughput (smoothed)")
    style_axis(ax, title="Training Throughput vs Step", ylabel="k tokens / sec")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "throughput.png"), dpi=150)
    plt.close(fig)


def plot_resp_tok_acc(acc_df, outdir):
    if acc_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(acc_df["step"], acc_df["resp_tok_acc"], color=ACCENT["acc"],
            alpha=0.35, linewidth=1)
    ax.plot(acc_df["step"], smooth(acc_df["resp_tok_acc"]), color=ACCENT["acc"],
            linewidth=2, label="Response token accuracy (smoothed)")
    style_axis(ax, title="Response Token Accuracy vs Step", ylabel="Accuracy (%)")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "resp_tok_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_dashboard(train_df, val_df, acc_df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(train_df["step"], train_df["loss"], color=ACCENT["train"], alpha=0.3, linewidth=1)
    ax.plot(train_df["step"], smooth(train_df["loss"]), color=ACCENT["train"], linewidth=2, label="Train")
    ax.plot(val_df["step"], val_df["val_loss"], color=ACCENT["val"], marker="o", markersize=4, linewidth=2, label="Val")
    style_axis(ax, title="Loss", ylabel="Loss")
    ax.legend(framealpha=0.9)

    ax = axes[0, 1]
    ax.plot(train_df["step"], train_df["lr"], color=ACCENT["lr"], linewidth=2)
    style_axis(ax, title="Learning Rate", ylabel="LR")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axes[1, 0]
    ax.plot(train_df["step"], train_df["grad_norm"], color=ACCENT["gnorm"], alpha=0.3, linewidth=1)
    ax.plot(train_df["step"], smooth(train_df["grad_norm"]), color=ACCENT["gnorm"], linewidth=2)
    style_axis(ax, title="Gradient Norm", ylabel="Grad norm")

    ax = axes[1, 1]
    if not acc_df.empty:
        ax.plot(acc_df["step"], acc_df["resp_tok_acc"], color=ACCENT["acc"], alpha=0.3, linewidth=1)
        ax.plot(acc_df["step"], smooth(acc_df["resp_tok_acc"]), color=ACCENT["acc"], linewidth=2)
        style_axis(ax, title="Response Token Accuracy", ylabel="Accuracy (%)")
    else:
        ax.plot(train_df["step"], train_df["tok_per_sec"] / 1000, color=ACCENT["tokspeed"], linewidth=2)
        style_axis(ax, title="Throughput", ylabel="k tok/s")

    fig.suptitle("Training Dashboard", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(outdir, "dashboard.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics (dark theme).")
    parser.add_argument("--csv", default="metrics.csv", help="Path to metrics.csv")
    parser.add_argument("--log", default="train.log", help="Path to train.log")
    parser.add_argument("--outdir", default="plots", help="Output directory for PNGs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    train_df, val_df = load_metrics_csv(args.csv)
    acc_df = load_resp_tok_acc(args.log) if os.path.exists(args.log) else pd.DataFrame(columns=["step", "resp_tok_acc"])

    plot_loss(train_df, val_df, args.outdir)
    plot_lr(train_df, args.outdir)
    plot_grad_norm(train_df, args.outdir)
    plot_throughput(train_df, args.outdir)
    plot_resp_tok_acc(acc_df, args.outdir)
    plot_dashboard(train_df, val_df, acc_df, args.outdir)

    print(f"Saved plots to: {args.outdir}/")
    for f in sorted(os.listdir(args.outdir)):
        print(" -", f)


if __name__ == "__main__":
    main()
