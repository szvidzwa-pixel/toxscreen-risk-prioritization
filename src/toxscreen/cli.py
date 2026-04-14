from __future__ import annotations

import argparse
import json

from .pipeline import run_audit, run_training


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="ToxScreen risk prioritization pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Profile the dataset and generate EDA outputs.",
    )
    audit_parser.add_argument("--data", required=True, help="Path to the input CSV.")

    train_parser = subparsers.add_parser("train", help="Train the ML pipeline.")
    train_parser.add_argument("--data", required=True, help="Path to the input CSV.")
    train_parser.add_argument("--config", required=True, help="Path to the JSON config.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        profile = run_audit(args.data)
        print(json.dumps(profile, indent=2))
        return

    if args.command == "train":
        results = run_training(args.data, args.config)
        print(json.dumps(results, indent=2))
