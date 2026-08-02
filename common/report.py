"""Console + markdown reporting helpers shared by every lab.

Each lab ends by calling `print_table()` (to show a comparison on screen),
`save_lab_results()` (so later labs / the capstone can load its numbers),
and `append_summary()` (to build up results/SUMMARY.md across the whole
suite).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config

# (raw key, display header, format spec)
DEFAULT_COLUMNS = [
    ("label", "Variant", "{}"),
    ("request_throughput", "Req/s", "{:.2f}"),
    ("output_throughput", "Out tok/s", "{:.1f}"),
    ("total_token_throughput", "Total tok/s", "{:.1f}"),
    ("mean_ttft_ms", "Mean TTFT ms", "{:.1f}"),
    ("p99_ttft_ms", "P99 TTFT ms", "{:.1f}"),
    ("mean_tpot_ms", "Mean TPOT ms", "{:.2f}"),
]


def _fmt(value, fmt) -> str:
    if value is None:
        return "-"
    try:
        return fmt.format(value)
    except (ValueError, TypeError):
        return str(value)


def print_table(rows: list[dict], columns=DEFAULT_COLUMNS, title: str | None = None) -> str:
    """Print (and return, as text) a simple fixed-width comparison table."""
    lines = []
    if title:
        lines.append(f"\n=== {title} ===")
    if not rows:
        lines.append("(no rows)")
        text = "\n".join(lines)
        print(text)
        return text

    headers = [c[1] for c in columns]
    keys = [c[0] for c in columns]
    fmts = [c[2] for c in columns]
    cells = [[_fmt(r.get(k), f) for k, f in zip(keys, fmts)] for r in rows]
    widths = [max(len(h), *(len(row[i]) for row in cells)) for i, h in enumerate(headers)]

    def fmt_row(vals: list[str]) -> str:
        return "  ".join(v.ljust(w) for v, w in zip(vals, widths))

    lines.append(fmt_row(headers))
    lines.append(fmt_row(["-" * w for w in widths]))
    for row in cells:
        lines.append(fmt_row(row))
    text = "\n".join(lines)
    print(text)
    return text


def add_speedup(rows: list[dict], baseline_label: str, metric: str = "output_throughput",
                 key: str = "speedup") -> list[dict]:
    """Return a copy of rows with a `key` column = metric / baseline metric."""
    baseline = next((r for r in rows if r.get("label") == baseline_label), None)
    out = []
    for r in rows:
        r2 = dict(r)
        base_val = baseline.get(metric) if baseline else None
        val = r.get(metric)
        r2[key] = (val / base_val) if (base_val and val is not None) else None
        out.append(r2)
    return out


def save_lab_results(lab_name: str, rows: list[dict]) -> Path:
    out_dir = config.RESULTS_DIR / lab_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"[report] saved {out_path}")
    return out_path


def load_lab_results(lab_name: str) -> list[dict] | None:
    path = config.RESULTS_DIR / lab_name / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_baseline(lab_name: str = "lab00_fp32_baseline", label: str | None = None) -> dict | None:
    """Load a specific saved row from a previous lab (default: lab00's fp32 baseline)."""
    rows = load_lab_results(lab_name)
    if not rows:
        return None
    if label is None:
        return rows[0]
    return next((r for r in rows if r.get("label") == label), rows[0])


def append_summary(lab_name: str, title: str, table_text: str, notes: str = "") -> None:
    config.SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not config.SUMMARY_PATH.exists()
    with open(config.SUMMARY_PATH, "a") as f:
        if is_new:
            f.write("# vLLM Optimization Labs - Results Summary\n")
        f.write(f"\n## {title}\n\n")
        f.write("```\n" + table_text + "\n```\n")
        if notes:
            f.write(f"\n{notes}\n")
    print(f"[report] appended '{title}' to {config.SUMMARY_PATH}")
