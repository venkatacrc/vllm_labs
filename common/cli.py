"""Tiny argparse helper so every lab's standalone entrypoint looks the same."""
from __future__ import annotations

import argparse


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="Shrink the workload for a fast smoke test.")
    p.add_argument("--gpus", default=None, help="GPU id(s) to use, e.g. '0' or '0,1,2,3'. Default: lab-specific.")
    return p
