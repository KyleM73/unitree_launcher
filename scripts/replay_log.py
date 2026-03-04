#!/usr/bin/env python3
"""Replay and inspect logged robot data.

Usage:
    uv run python scripts/replay_log.py logs/<run>/ [--format csv] [--output file]
"""
import argparse
from pathlib import Path

from unitree_launcher.datalog.replay import LogReplay


def main():
    parser = argparse.ArgumentParser(description="Replay and inspect logged robot data")
    parser.add_argument("log_dir", help="Path to log run directory")
    parser.add_argument("--format", choices=["csv", "summary"], default="summary",
                        help="Output format (default: summary)")
    parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    args = parser.parse_args()

    replay = LogReplay(args.log_dir)
    replay.load()

    if args.format == "csv":
        output_path = args.output or str(Path(args.log_dir) / "export.csv")
        replay.to_csv(output_path)
        print(f"Exported {replay.n_steps} steps to {output_path}")
    else:
        text = replay.summary()
        if args.output:
            with open(args.output, "w") as f:
                f.write(text + "\n")
        else:
            print(text)


if __name__ == "__main__":
    main()
