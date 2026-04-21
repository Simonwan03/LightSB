import argparse
import os
import sys
from typing import Any, Dict

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.ffhq_metadata import build_ffhq_metadata, write_metadata_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build unified FFHQ metadata from DCGM JSON files and local npy labels."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/ffhq_metadata_integration.yaml",
        help="Path to metadata integration YAML config.",
    )
    parser.add_argument("--latents-path", default=None, help="Override latents_path.")
    parser.add_argument("--age-path", default=None, help="Override age_path.")
    parser.add_argument("--gender-path", default=None, help="Override gender_path.")
    parser.add_argument("--json-dir", default=None, help="Override ffhq_json_dir.")
    parser.add_argument("--output", default=None, help="Override output_metadata_path.")
    parser.add_argument(
        "--trust-json-age-gender",
        action="store_true",
        help="Use JSON gender/age as canonical labels instead of existing npy labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    apply_overrides(config, args)
    resolve_config_paths(config)

    rows, report = build_ffhq_metadata(config)
    write_metadata_outputs(
        rows,
        csv_path=config["output_metadata_path"],
        report=report,
        report_path=config.get("output_report_path"),
        pkl_path=config.get("output_pickle_path"),
        parquet_path=config.get("output_parquet_path"),
    )

    print(f"Wrote metadata CSV: {config['output_metadata_path']}")
    if config.get("output_pickle_path"):
        print(f"Wrote metadata pickle: {config['output_pickle_path']}")
    if config.get("output_report_path"):
        print(f"Wrote validation report: {config['output_report_path']}")
    print_report(report.to_dict())


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.latents_path:
        config["latents_path"] = args.latents_path
    if args.age_path:
        config["age_path"] = args.age_path
    if args.gender_path:
        config["gender_path"] = args.gender_path
    if args.json_dir:
        config["ffhq_json_dir"] = args.json_dir
    if args.output:
        config["output_metadata_path"] = args.output
    if args.trust_json_age_gender:
        config["trust_existing_age_gender"] = False


def resolve_config_paths(config: Dict[str, Any]) -> None:
    path_keys = [
        "latents_path",
        "age_path",
        "gender_path",
        "ffhq_json_dir",
        "output_metadata_path",
        "output_report_path",
        "output_pickle_path",
        "output_parquet_path",
    ]
    for key in path_keys:
        if config.get(key):
            config[key] = resolve_path(config[key])


def print_report(report: Dict[str, Any]) -> None:
    print("Validation summary:")
    for key in [
        "latent_count",
        "json_records_loaded",
        "expected_json_count",
        "missing_json_count",
        "malformed_json_count",
        "gender_compared_count",
        "gender_mismatch_count",
        "age_compared_count",
        "age_mismatch_count",
    ]:
        print(f"  {key}: {report.get(key)}")
    if report.get("missing_json_examples"):
        print(f"  missing_json_examples: {report['missing_json_examples'][:10]}")
    if report.get("gender_mismatch_examples"):
        print(f"  gender_mismatch_examples: {report['gender_mismatch_examples'][:3]}")
    if report.get("age_mismatch_examples"):
        print(f"  age_mismatch_examples: {report['age_mismatch_examples'][:3]}")


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(REPO_ROOT, path)


if __name__ == "__main__":
    main()
