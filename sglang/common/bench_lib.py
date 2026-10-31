#!/usr/bin/env python3
"""
Shared helper for parsing `python -m sglang.bench_serving` output and
building a running summary.csv across all labs so lab11 (the capstone) can
produce a single before/after comparison table and chart.

`sglang.bench_serving` writes one JSON object per invocation to the file
passed via --output-file (JSON-lines format). The exact set of keys has
changed a bit across SGLang releases, so we parse defensively: pull out the
metrics we care about if present, and otherwise leave them blank rather than
crashing the lab.

Usage:
    python bench_lib.py record --lab lab01 --config "fp32-baseline" \
        --jsonl results/lab01/bench.jsonl --summary-csv results/summary.csv \
        [--extra key=value ...]

    python bench_lib.py show --summary-csv results/summary.csv
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

# Metrics we try to pull out of a bench_serving result record. Multiple
# candidate key names are listed because they vary slightly across versions.
METRIC_CANDIDATES = {
    "request_throughput_rps": ["request_throughput"],
    "input_throughput_tok_s": ["input_throughput"],
    "output_throughput_tok_s": ["output_throughput"],
    "total_throughput_tok_s": ["total_token_throughput", "total_throughput"],
    "mean_ttft_ms": ["mean_ttft_ms"],
    "median_ttft_ms": ["median_ttft_ms"],
    "p99_ttft_ms": ["p99_ttft_ms"],
    "mean_tpot_ms": ["mean_tpot_ms"],
    "median_tpot_ms": ["median_tpot_ms"],
    "p99_tpot_ms": ["p99_tpot_ms"],
    "mean_itl_ms": ["mean_itl_ms"],
    "median_itl_ms": ["median_itl_ms"],
    "completed": ["completed"],
    "duration_s": ["duration"],
}

CSV_FIELDS = [
    "timestamp",
    "lab",
    "config",
    *METRIC_CANDIDATES.keys(),
    "extra",
    "source_jsonl",
]


def _last_json_object(path: str) -> dict:
    """bench_serving appends one JSON object per line; take the most recent."""
    last = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                # Some sglang versions pretty-print a single JSON blob instead
                # of jsonl. Fall back to parsing the whole file at the end.
                continue
    if last is None:
        with open(path, "r") as f:
            content = f.read().strip()
        if content:
            last = json.loads(content)
    if last is None:
        raise ValueError(f"Could not parse any JSON out of {path}")
    return last


def extract_metrics(record: dict) -> dict:
    out = {}
    for canonical, candidates in METRIC_CANDIDATES.items():
        val = None
        for c in candidates:
            if c in record:
                val = record[c]
                break
        out[canonical] = val
    return out


def cmd_record(args):
    record = _last_json_object(args.jsonl)
    metrics = extract_metrics(record)

    extra = {}
    for kv in args.extra or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        extra[k] = v

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lab": args.lab,
        "config": args.config,
        **metrics,
        "extra": json.dumps(extra) if extra else "",
        "source_jsonl": os.path.abspath(args.jsonl),
    }

    write_header = not os.path.exists(args.summary_csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.summary_csv)), exist_ok=True)
    with open(args.summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Recorded [{args.lab}] {args.config}:")
    for k in ("total_throughput_tok_s", "output_throughput_tok_s", "median_ttft_ms", "median_tpot_ms"):
        if metrics.get(k) is not None:
            print(f"  {k:28s} = {metrics[k]}")


def cmd_record_raw(args):
    """Like `record`, but takes metrics directly as key=value pairs instead of
    parsing a bench_serving jsonl file. Useful for non-SGLang baselines (e.g.
    the naive HuggingFace transformers loop in lab01) that report metrics in
    their own JSON format.
    """
    metrics = {k: None for k in METRIC_CANDIDATES}
    for kv in args.metric or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if k in metrics:
            try:
                v = float(v)
            except ValueError:
                pass
            metrics[k] = v

    extra = {}
    for kv in args.extra or []:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        extra[k] = v

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lab": args.lab,
        "config": args.config,
        **metrics,
        "extra": json.dumps(extra) if extra else "",
        "source_jsonl": args.source or "",
    }
    write_header = not os.path.exists(args.summary_csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.summary_csv)), exist_ok=True)
    with open(args.summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Recorded [{args.lab}] {args.config} (raw metrics)")


def cmd_show(args):
    if not os.path.exists(args.summary_csv):
        print(f"No summary yet at {args.summary_csv}")
        return
    with open(args.summary_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("summary.csv is empty")
        return
    cols = ["lab", "config", "total_throughput_tok_s", "output_throughput_tok_s",
            "median_ttft_ms", "median_tpot_ms"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_record = sub.add_parser("record", help="Parse a bench_serving output file and append to summary.csv")
    p_record.add_argument("--lab", required=True)
    p_record.add_argument("--config", required=True)
    p_record.add_argument("--jsonl", required=True)
    p_record.add_argument("--summary-csv", required=True)
    p_record.add_argument("--extra", action="append", help="key=value, repeatable")
    p_record.set_defaults(func=cmd_record)

    p_raw = sub.add_parser("record-raw", help="Append a row to summary.csv from explicit key=value metrics")
    p_raw.add_argument("--lab", required=True)
    p_raw.add_argument("--config", required=True)
    p_raw.add_argument("--metric", action="append", help="key=value, repeatable (keys: " + ", ".join(METRIC_CANDIDATES) + ")")
    p_raw.add_argument("--extra", action="append", help="key=value, repeatable")
    p_raw.add_argument("--source", default="")
    p_raw.add_argument("--summary-csv", required=True)
    p_raw.set_defaults(func=cmd_record_raw)

    p_show = sub.add_parser("show", help="Pretty-print summary.csv")
    p_show.add_argument("--summary-csv", required=True)
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
