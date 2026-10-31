#!/usr/bin/env python3
"""
Render results/summary.csv (populated incrementally by every lab via
common/bench_lib.py) into a set of bar charts under results/plots/, and print
a human-readable comparison table.

Usage:
    python plot_results.py [--summary-csv results/summary.csv] [--out-dir results/plots]

This is intentionally dependency-light (matplotlib + stdlib csv) so it runs
inside the same sglang-labs conda env without extra installs. If matplotlib
isn't available it just prints the table and skips the charts.
"""
import argparse
import csv
import os


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def print_table(rows):
    if not rows:
        print("No results recorded yet. Run some labs first!")
        return
    cols = ["lab", "config", "total_throughput_tok_s", "output_throughput_tok_s",
            "median_ttft_ms", "median_tpot_ms"]
    widths = {c: max(len(c), max((len(str(r.get(c, "") or "")) for r in rows), default=0)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r.get(c, "") or "").ljust(widths[c]) for c in cols))


def make_charts(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts (pip install matplotlib to enable).")
        return

    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        ("output_throughput_tok_s", "Output throughput (tok/s, higher is better)", "throughput.png"),
        ("median_ttft_ms", "Median TTFT (ms, lower is better)", "ttft.png"),
        ("median_tpot_ms", "Median TPOT (ms/token, lower is better)", "tpot.png"),
    ]

    for metric, title, fname in metrics:
        labels, values = [], []
        for r in rows:
            v = to_float(r.get(metric))
            if v is None:
                continue
            labels.append(f"{r['lab']}\n{r['config']}")
            values.append(v)
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 5))
        bars = ax.bar(range(len(labels)), values, color="#4C72B0")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(metric)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default=os.path.join(os.path.dirname(__file__), "..", "results", "summary.csv"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "results", "plots"))
    args = ap.parse_args()

    rows = load_rows(args.summary_csv)
    print_table(rows)
    make_charts(rows, args.out_dir)


if __name__ == "__main__":
    main()
