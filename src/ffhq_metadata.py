import csv
import json
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


MetadataRows = List[Dict[str, Any]]
Constraints = Mapping[str, Any]


DEFAULT_COLUMNS = [
    "index",
    "latent_index",
    "gender",
    "age",
    "age_group",
    "glasses",
    "smile",
    "pitch",
    "roll",
    "yaw",
    "moustache",
    "beard",
    "sideburns",
    "has_beard",
    "has_facial_hair",
    "blur_level",
    "blur_value",
    "exposure_level",
    "exposure_value",
    "noise_level",
    "noise_value",
    "eye_makeup",
    "lip_makeup",
    "has_makeup",
    "forehead_occluded",
    "eye_occluded",
    "mouth_occluded",
    "has_occlusion",
    "bald",
    "hair_invisible",
    "dominant_hair_color",
    "dominant_hair_confidence",
    "accessories_count",
    "has_accessories",
    "emotion_anger",
    "emotion_contempt",
    "emotion_disgust",
    "emotion_fear",
    "emotion_happiness",
    "emotion_neutral",
    "emotion_sadness",
    "emotion_surprise",
    "existing_gender",
    "json_gender",
    "existing_age",
    "json_age",
    "gender_mismatch",
    "age_abs_diff",
    "json_path",
    "json_missing",
    "json_malformed",
]


@dataclass
class MetadataBuildReport:
    latent_count: int
    json_records_loaded: int
    expected_json_count: Optional[int]
    missing_json_count: int
    malformed_json_count: int
    gender_compared_count: int
    gender_mismatch_count: int
    age_compared_count: int
    age_mismatch_count: int
    missing_json_examples: List[int]
    malformed_json_examples: List[str]
    gender_mismatch_examples: List[Dict[str, Any]]
    age_mismatch_examples: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def build_ffhq_metadata(config: Mapping[str, Any]) -> Tuple[MetadataRows, MetadataBuildReport]:
    latents_path = config["latents_path"]
    age_path = config.get("age_path")
    gender_path = config.get("gender_path")
    json_dir = config["ffhq_json_dir"]

    latents = np.load(latents_path, mmap_mode="r")
    latent_count = int(latents.shape[0])
    existing_age = load_optional_array(age_path)
    existing_gender = load_optional_array(gender_path)
    if existing_age is not None:
        validate_label_length("age", existing_age, latent_count)
    if existing_gender is not None:
        validate_label_length("gender", existing_gender, latent_count)

    json_by_index, malformed = load_ffhq_json_dir(json_dir)
    expected_json_count = config.get("expected_json_count", latent_count)
    fail_on_missing = bool(config.get("fail_on_missing_json", False))
    missing = [index for index in range(latent_count) if index not in json_by_index]
    if fail_on_missing and missing:
        raise ValueError(
            f"Missing {len(missing)} FFHQ JSON files. Examples: {missing[:10]}"
        )
    if expected_json_count is not None and len(json_by_index) != int(expected_json_count):
        print(
            "Warning: loaded "
            f"{len(json_by_index)} JSON records, expected {expected_json_count}."
        )

    trust_existing = bool(config.get("trust_existing_age_gender", True))
    age_bins = config.get("age_bins") or default_age_bins()
    age_tolerance = float(config.get("age_mismatch_tolerance", 1.0))
    rows = []

    gender_mismatches = []
    age_mismatches = []
    gender_compared = 0
    age_compared = 0

    for latent_index in range(latent_count):
        raw_json = json_by_index.get(latent_index)
        parsed = parse_ffhq_json_record(raw_json) if raw_json is not None else empty_json_features()
        old_gender = normalize_gender(existing_gender[latent_index]) if existing_gender is not None else "unknown"
        old_age = normalize_number(existing_age[latent_index]) if existing_age is not None else None
        json_gender = parsed["json_gender"]
        json_age = parsed["json_age"]

        gender = old_gender if trust_existing and old_gender != "unknown" else json_gender
        age = old_age if trust_existing and old_age is not None and old_age >= 0 else json_age

        gender_mismatch = False
        if old_gender != "unknown" and json_gender != "unknown":
            gender_compared += 1
            gender_mismatch = old_gender != json_gender
            if gender_mismatch and len(gender_mismatches) < 20:
                gender_mismatches.append(
                    {
                        "index": latent_index,
                        "existing_gender": old_gender,
                        "json_gender": json_gender,
                    }
                )

        age_abs_diff = ""
        age_mismatch = False
        if old_age is not None and old_age >= 0 and json_age is not None and json_age >= 0:
            age_compared += 1
            age_abs_diff = abs(float(old_age) - float(json_age))
            age_mismatch = age_abs_diff > age_tolerance
            if age_mismatch and len(age_mismatches) < 20:
                age_mismatches.append(
                    {
                        "index": latent_index,
                        "existing_age": old_age,
                        "json_age": json_age,
                        "age_abs_diff": age_abs_diff,
                    }
                )

        row = {
            **parsed,
            "index": latent_index,
            "latent_index": latent_index,
            "gender": gender,
            "age": "" if age is None else age,
            "age_group": age_to_group(age, age_bins),
            "existing_gender": old_gender,
            "json_gender": json_gender,
            "existing_age": "" if old_age is None else old_age,
            "json_age": "" if json_age is None else json_age,
            "gender_mismatch": gender_mismatch,
            "age_abs_diff": age_abs_diff,
            "json_path": os.path.join(json_dir, f"{latent_index:05d}.json"),
            "json_missing": raw_json is None,
            "json_malformed": False,
        }
        rows.append(row)

    malformed_indices = []
    for path, error in malformed:
        index = parse_json_index(path)
        if index is not None and 0 <= index < latent_count:
            rows[index]["json_malformed"] = True
            malformed_indices.append(index)

    report = MetadataBuildReport(
        latent_count=latent_count,
        json_records_loaded=len(json_by_index),
        expected_json_count=None if expected_json_count is None else int(expected_json_count),
        missing_json_count=len(missing),
        malformed_json_count=len(malformed),
        gender_compared_count=gender_compared,
        gender_mismatch_count=sum(bool(row["gender_mismatch"]) for row in rows),
        age_compared_count=age_compared,
        age_mismatch_count=sum(
            isinstance(row["age_abs_diff"], float) and row["age_abs_diff"] > age_tolerance
            for row in rows
        ),
        missing_json_examples=missing[:20],
        malformed_json_examples=[f"{path}: {error}" for path, error in malformed[:20]],
        gender_mismatch_examples=gender_mismatches,
        age_mismatch_examples=age_mismatches,
    )
    return rows, report


def load_ffhq_json_dir(json_dir: str) -> Tuple[Dict[int, Dict[str, Any]], List[Tuple[str, str]]]:
    if not os.path.isdir(json_dir):
        raise FileNotFoundError(f"FFHQ JSON directory not found: {json_dir}")

    records: Dict[int, Dict[str, Any]] = {}
    malformed = []
    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(json_dir, filename)
        index = parse_json_index(filename)
        if index is None:
            malformed.append((path, "filename does not contain an integer index"))
            continue
        try:
            with open(path) as handle:
                payload = json.load(handle)
            records[index] = payload
        except Exception as exc:
            malformed.append((path, str(exc)))
    return records, malformed


def parse_ffhq_json_record(record: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not record:
        return empty_json_features()
    if isinstance(record, list):
        record = record[0] if record else {}
    attrs = dict(record.get("faceAttributes") or {})

    head_pose = attrs.get("headPose") or {}
    facial_hair = attrs.get("facialHair") or {}
    blur = attrs.get("blur") or {}
    exposure = attrs.get("exposure") or {}
    noise = attrs.get("noise") or {}
    makeup = attrs.get("makeup") or {}
    occlusion = attrs.get("occlusion") or {}
    hair = attrs.get("hair") or {}
    emotion = attrs.get("emotion") or {}
    accessories = attrs.get("accessories") or []

    moustache = normalize_number(facial_hair.get("moustache"))
    beard = normalize_number(facial_hair.get("beard"))
    sideburns = normalize_number(facial_hair.get("sideburns"))
    dominant_color, dominant_confidence = dominant_hair_color(hair.get("hairColor") or [])
    forehead_occluded = normalize_bool(occlusion.get("foreheadOccluded"))
    eye_occluded = normalize_bool(occlusion.get("eyeOccluded"))
    mouth_occluded = normalize_bool(occlusion.get("mouthOccluded"))
    eye_makeup = normalize_bool(makeup.get("eyeMakeup"))
    lip_makeup = normalize_bool(makeup.get("lipMakeup"))

    return {
        "json_gender": normalize_gender(attrs.get("gender")),
        "json_age": normalize_number(attrs.get("age")),
        "glasses": normalize_glasses(attrs.get("glasses")),
        "smile": normalize_number(attrs.get("smile")),
        "pitch": normalize_number(head_pose.get("pitch")),
        "roll": normalize_number(head_pose.get("roll")),
        "yaw": normalize_number(head_pose.get("yaw")),
        "moustache": moustache,
        "beard": beard,
        "sideburns": sideburns,
        "has_beard": threshold_bool(beard),
        "has_facial_hair": any(threshold_bool(v) for v in (moustache, beard, sideburns)),
        "blur_level": normalize_text(blur.get("blurLevel")),
        "blur_value": normalize_number(blur.get("value")),
        "exposure_level": normalize_text(exposure.get("exposureLevel")),
        "exposure_value": normalize_number(exposure.get("value")),
        "noise_level": normalize_text(noise.get("noiseLevel")),
        "noise_value": normalize_number(noise.get("value")),
        "eye_makeup": eye_makeup,
        "lip_makeup": lip_makeup,
        "has_makeup": bool(eye_makeup or lip_makeup),
        "forehead_occluded": forehead_occluded,
        "eye_occluded": eye_occluded,
        "mouth_occluded": mouth_occluded,
        "has_occlusion": bool(forehead_occluded or eye_occluded or mouth_occluded),
        "bald": normalize_number(hair.get("bald")),
        "hair_invisible": normalize_bool(hair.get("invisible")),
        "dominant_hair_color": dominant_color,
        "dominant_hair_confidence": dominant_confidence,
        "accessories_count": len(accessories) if isinstance(accessories, list) else 0,
        "has_accessories": bool(accessories),
        "emotion_anger": normalize_number(emotion.get("anger")),
        "emotion_contempt": normalize_number(emotion.get("contempt")),
        "emotion_disgust": normalize_number(emotion.get("disgust")),
        "emotion_fear": normalize_number(emotion.get("fear")),
        "emotion_happiness": normalize_number(emotion.get("happiness")),
        "emotion_neutral": normalize_number(emotion.get("neutral")),
        "emotion_sadness": normalize_number(emotion.get("sadness")),
        "emotion_surprise": normalize_number(emotion.get("surprise")),
    }


def empty_json_features() -> Dict[str, Any]:
    row = {
        "json_gender": "unknown",
        "json_age": None,
        "glasses": "unknown",
        "smile": "",
        "pitch": "",
        "roll": "",
        "yaw": "",
        "moustache": "",
        "beard": "",
        "sideburns": "",
        "has_beard": False,
        "has_facial_hair": False,
        "blur_level": "unknown",
        "blur_value": "",
        "exposure_level": "unknown",
        "exposure_value": "",
        "noise_level": "unknown",
        "noise_value": "",
        "eye_makeup": False,
        "lip_makeup": False,
        "has_makeup": False,
        "forehead_occluded": False,
        "eye_occluded": False,
        "mouth_occluded": False,
        "has_occlusion": False,
        "bald": "",
        "hair_invisible": False,
        "dominant_hair_color": "unknown",
        "dominant_hair_confidence": "",
        "accessories_count": 0,
        "has_accessories": False,
        "emotion_anger": "",
        "emotion_contempt": "",
        "emotion_disgust": "",
        "emotion_fear": "",
        "emotion_happiness": "",
        "emotion_neutral": "",
        "emotion_sadness": "",
        "emotion_surprise": "",
    }
    return row


def write_metadata_outputs(
    rows: MetadataRows,
    csv_path: str,
    report: Optional[MetadataBuildReport] = None,
    report_path: Optional[str] = None,
    pkl_path: Optional[str] = None,
    parquet_path: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ordered_fieldnames(rows)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if pkl_path:
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, "wb") as handle:
            pickle.dump(rows, handle)

    if parquet_path:
        try:
            import pandas as pd

            os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
            pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        except Exception as exc:
            print(f"Warning: could not write parquet output {parquet_path}: {exc}")

    if report and report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as handle:
            json.dump(report.to_dict(), handle, indent=2)


def read_metadata_table(path: str):
    try:
        import pandas as pd

        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        with open(path, newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]


def filter_metadata(df: Any, constraints: Optional[Constraints] = None):
    if not constraints:
        return df.copy() if hasattr(df, "copy") else list(df)
    if is_pandas_dataframe(df):
        mask = None
        for column, expected in constraints.items():
            current = pandas_constraint_mask(df, column, expected)
            mask = current if mask is None else (mask & current)
        return df[mask].copy()
    return [row for row in df if row_matches(row, constraints)]


def select_subset(df: Any, **constraints: Any):
    return filter_metadata(df, constraints)


def build_source_target_subsets(
    df: Any,
    source_filter: Constraints,
    target_filter: Constraints,
) -> Tuple[Any, Any]:
    return filter_metadata(df, source_filter), filter_metadata(df, target_filter)


def metadata_statistics(df: Any, intersections: Optional[Sequence[Constraints]] = None) -> Dict[str, Any]:
    rows = dataframe_to_rows(df)
    stats: Dict[str, Any] = {}
    for column in ("gender", "age_group", "glasses"):
        stats[column] = dict(Counter(str(row.get(column, "")) for row in rows))
    stats["intersections"] = []
    for constraints in intersections or []:
        subset = [row for row in rows if row_matches(row, constraints)]
        stats["intersections"].append({"filter": dict(constraints), "count": len(subset)})
    return stats


def default_intersections() -> List[Constraints]:
    return [
        {"gender": "male", "age_group": "adult", "glasses": "no_glasses"},
        {"gender": "female", "age_group": "adult", "glasses": "no_glasses"},
        {"gender": "female", "age_group": "child", "glasses": ["reading_glasses", "sunglasses"]},
        {"gender": "female", "age_group": "child", "glasses": "no_glasses"},
    ]


def normalize_gender(value: Any) -> str:
    text = normalize_text(value)
    if text in {"m", "man", "male", "1"}:
        return "male"
    if text in {"f", "woman", "female", "0"}:
        return "female"
    return text if text else "unknown"


def normalize_glasses(value: Any) -> str:
    text = normalize_text(value)
    mapping = {
        "noglasses": "no_glasses",
        "no_glasses": "no_glasses",
        "none": "no_glasses",
        "0": "no_glasses",
        "false": "no_glasses",
        "readingglasses": "reading_glasses",
        "reading_glasses": "reading_glasses",
        "glasses": "reading_glasses",
        "1": "reading_glasses",
        "true": "reading_glasses",
        "sunglasses": "sunglasses",
        "sun_glasses": "sunglasses",
        "swimminggoggles": "swimming_goggles",
        "swimming_goggles": "swimming_goggles",
    }
    return mapping.get(text, text or "unknown")


def age_to_group(age: Any, age_bins: Sequence[Mapping[str, Any]]) -> str:
    value = normalize_number(age)
    if value is None or value < 0:
        return "unknown"
    for bin_config in age_bins:
        lower = bin_config.get("min", float("-inf"))
        upper = bin_config.get("max", float("inf"))
        if value >= float(lower) and value < float(upper):
            return str(bin_config["name"])
    return "unknown"


def default_age_bins() -> List[Dict[str, Any]]:
    return [
        {"name": "child", "min": 0, "max": 13},
        {"name": "teen", "min": 13, "max": 18},
        {"name": "young_adult", "min": 18, "max": 30},
        {"name": "adult", "min": 30, "max": 60},
        {"name": "senior", "min": 60, "max": 200},
    ]


def dominant_hair_color(hair_colors: Sequence[Mapping[str, Any]]) -> Tuple[str, Any]:
    if not hair_colors:
        return "unknown", ""
    best = max(hair_colors, key=lambda item: normalize_number(item.get("confidence")) or -1)
    return normalize_text(best.get("color")) or "unknown", normalize_number(best.get("confidence"))


def row_matches(row: Mapping[str, Any], constraints: Constraints) -> bool:
    for key, expected in constraints.items():
        if key not in row:
            return False
        actual = row[key]
        if isinstance(expected, Mapping):
            value = normalize_number(actual)
            if value is None:
                return False
            if "min" in expected and value < float(expected["min"]):
                return False
            if "max" in expected and value >= float(expected["max"]):
                return False
        elif isinstance(expected, (list, tuple, set)):
            if not any(values_equal(actual, item) for item in expected):
                return False
        elif not values_equal(actual, expected):
            return False
    return True


def pandas_constraint_mask(df: Any, column: str, expected: Any):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in metadata table.")
    if isinstance(expected, Mapping):
        values = df[column].astype(float)
        mask = values.notna()
        if "min" in expected:
            mask = mask & (values >= float(expected["min"]))
        if "max" in expected:
            mask = mask & (values < float(expected["max"]))
        return mask
    if isinstance(expected, (list, tuple, set)):
        expected_values = {normalize_text(item) for item in expected}
        return df[column].astype(str).map(normalize_text).isin(expected_values)
    return df[column].astype(str).map(normalize_text) == normalize_text(expected)


def dataframe_to_rows(df: Any) -> MetadataRows:
    if is_pandas_dataframe(df):
        return df.to_dict(orient="records")
    return list(df)


def ordered_fieldnames(rows: MetadataRows) -> List[str]:
    keys = {key for row in rows for key in row.keys()}
    ordered = [column for column in DEFAULT_COLUMNS if column in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def values_equal(actual: Any, expected: Any) -> bool:
    return normalize_text(actual) == normalize_text(expected)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            value = " ".join(map(str, value.tolist()))
    elif hasattr(value, "item"):
        value = value.item()
    text = str(value).strip().lower()
    return text.replace(" ", "_")


def normalize_number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = normalize_text(value)
    return text in {"true", "1", "yes", "y"}


def threshold_bool(value: Any, threshold: float = 0.5) -> bool:
    number = normalize_number(value)
    return bool(number is not None and number >= threshold)


def load_optional_array(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    return np.load(path, allow_pickle=True).reshape(-1)


def validate_label_length(name: str, values: np.ndarray, expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} has {len(values)} values, but latents has {expected} rows.")


def parse_json_index(path_or_name: str) -> Optional[int]:
    basename = os.path.basename(path_or_name)
    stem = os.path.splitext(basename)[0]
    try:
        return int(stem)
    except ValueError:
        return None


def is_pandas_dataframe(obj: Any) -> bool:
    return hasattr(obj, "columns") and hasattr(obj, "to_dict") and hasattr(obj, "__getitem__")
