import argparse
import json
import os
import sys
from typing import Any, Dict, List


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.ffhq_metadata import (  # noqa: E402
    default_intersections,
    metadata_statistics,
    read_metadata_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print FFHQ metadata subset statistics.")
    parser.add_argument(
        "--metadata",
        default="data/ffhq_metadata.csv",
        help="Path to unified metadata CSV/parquet.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print statistics as JSON instead of a readable table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = resolve_path(args.metadata)
    table = read_metadata_table(metadata_path)
    stats = metadata_statistics(table, intersections=default_intersections())
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print_counts("gender", stats["gender"])
        print_counts("age_group", stats["age_group"])
        print_counts("glasses", stats["glasses"])
        print("intersections")
        for item in stats["intersections"]:
            print(f"  {format_filter(item['filter'])}: {item['count']}")


def print_counts(name: str, counts: Dict[str, int]) -> None:
    print(name)
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {key}: {value}")


def format_filter(filters: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key, value in filters.items():
        if isinstance(value, list):
            value = "|".join(map(str, value))
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(REPO_ROOT, path)


if __name__ == "__main__":
    main()
