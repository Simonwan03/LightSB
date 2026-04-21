import argparse
import csv
import os
import sys
from typing import Any, Optional

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compositional LightSB metadata from FFHQ latent/attribute arrays."
    )
    parser.add_argument("--latents", default="data/latents.npy", help="Path to latent matrix.")
    parser.add_argument("--gender", default="data/gender.npy", help="Path to gender labels.")
    parser.add_argument("--age", default="data/age.npy", help="Path to age labels.")
    parser.add_argument("--glasses", default=None, help="Optional path to glasses labels.")
    parser.add_argument("--expression", default=None, help="Optional path to expression labels.")
    parser.add_argument("--output", default="data/ffhq_attributes.csv", help="Output metadata CSV.")
    parser.add_argument(
        "--default-glasses",
        default="unknown",
        choices=("unknown", "yes", "no"),
        help="Value used when --glasses is not provided.",
    )
    parser.add_argument(
        "--default-expression",
        default="unknown",
        choices=("unknown", "neutral", "smiling", "other"),
        help="Value used when --expression is not provided.",
    )
    parser.add_argument(
        "--image-prefix",
        default="",
        help="Optional prefix for image_path values, e.g. data/ffhq_images.",
    )
    parser.add_argument(
        "--image-extension",
        default=".png",
        help="Image extension used with --image-prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latents_path = resolve_path(args.latents)
    gender_path = resolve_path(args.gender)
    age_path = resolve_path(args.age)
    output_path = resolve_path(args.output)

    latents = np.load(latents_path, mmap_mode="r")
    gender = np.load(gender_path, allow_pickle=True).reshape(-1)
    age = np.load(age_path, allow_pickle=True).reshape(-1)
    glasses = load_optional_labels(args.glasses)
    expression = load_optional_labels(args.expression)

    sample_count = int(latents.shape[0])
    validate_length("gender", gender, sample_count)
    validate_length("age", age, sample_count)
    if glasses is not None:
        validate_length("glasses", glasses, sample_count)
    if expression is not None:
        validate_length("expression", expression, sample_count)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "latent_path",
                "gender",
                "age_group",
                "age",
                "expression",
                "glasses",
            ],
        )
        writer.writeheader()
        for index in range(sample_count):
            writer.writerow(
                {
                    "image_path": make_image_path(args.image_prefix, args.image_extension, index),
                    "latent_path": "",
                    "gender": normalize_gender(gender[index]),
                    "age_group": age_to_group(age[index]),
                    "age": normalize_scalar(age[index]),
                    "expression": normalize_expression(
                        expression[index] if expression is not None else args.default_expression
                    ),
                    "glasses": normalize_glasses(
                        glasses[index] if glasses is not None else args.default_glasses
                    ),
                }
            )

    print(f"Wrote {sample_count} metadata rows to {output_path}")
    if glasses is None:
        print(
            "No glasses labels were provided. The glasses column was filled with "
            f"'{args.default_glasses}'. Use a gender/age-only config or provide --glasses "
            "before training a glasses bridge."
        )
    if expression is None:
        print(
            "No expression labels were provided. The expression column was filled with "
            f"'{args.default_expression}'. Provide --expression or use "
            "scripts/build_ffhq_metadata.py before training an expression bridge."
        )


def load_optional_labels(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    return np.load(resolve_path(path), allow_pickle=True).reshape(-1)


def validate_length(name: str, labels: np.ndarray, expected: int) -> None:
    if len(labels) != expected:
        raise ValueError(f"{name} has {len(labels)} labels, but latents has {expected} rows.")


def normalize_gender(value: Any) -> str:
    text = str(normalize_scalar(value)).strip().lower()
    if text in {"m", "man", "male", "1"}:
        return "male"
    if text in {"f", "woman", "female", "0"}:
        return "female"
    return text or "unknown"


def age_to_group(value: Any) -> str:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if age < 0:
        return "unknown"
    if age < 18:
        return "child"
    if age < 60:
        return "adult"
    return "senior"


def normalize_glasses(value: Any) -> str:
    text = str(normalize_scalar(value)).strip().lower()
    if text in {"1", "true", "yes", "y", "glasses"}:
        return "yes"
    if text in {"0", "false", "no", "n", "no_glasses", "none"}:
        return "no"
    return text or "unknown"


def normalize_expression(value: Any) -> str:
    text = str(normalize_scalar(value)).strip().lower().replace(" ", "_")
    if text in {"smile", "smiled", "smiling", "happy", "happiness", "1", "true", "yes"}:
        return "smiling"
    if text in {"neutral", "none", "0", "false", "no"}:
        return "neutral"
    if text in {"other", "unknown"}:
        return text
    return text or "unknown"


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return " ".join(map(str, value.tolist()))
    if hasattr(value, "item"):
        return value.item()
    return value


def make_image_path(prefix: str, extension: str, index: int) -> str:
    if not prefix:
        return ""
    return os.path.join(prefix, f"{index:06d}{extension}")


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


if __name__ == "__main__":
    main()
