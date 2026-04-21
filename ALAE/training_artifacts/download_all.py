import os
import sys
import urllib.request


ALAE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

WEIGHTS = {
    "ffhq": [
        (
            "170Qldnn28IwnVm9CQEq1AZhVsK7PJ0Xz",
            "https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_submitted.pth",
        ),
        (
            "1QESywJW8N-g3n0Csy0clztuJV99g8pRm",
            "https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_194.pth",
        ),
        (
            "18BzFYKS3icFd1DQKKTeje7CKbEKXPVug",
            "https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth",
        ),
    ],
    "celeba": [
        (
            "1T4gkE7-COHpX38qPwjMYO-xU-SrY_aT4",
            "https://alaeweights.s3.us-east-2.amazonaws.com/celeba/model_final.pth",
        ),
    ],
    "bedroom": [
        (
            "1gmYbc6Z8qJHJwICYDsB4aBMxXjnKeXA_",
            "https://alaeweights.s3.us-east-2.amazonaws.com/bedroom/model_final.pth",
        ),
    ],
    "celeba-hq256": [
        (
            "1ihJvp8iJWcLxTIjkV5cyA7l9TrxlUPkG",
            "https://alaeweights.s3.us-east-2.amazonaws.com/celeba-hq256/model_262r.pth",
        ),
        (
            "1gFQsGCNKo-frzKmA3aCvx07ShRymRIKZ",
            "https://alaeweights.s3.us-east-2.amazonaws.com/celeba-hq256/model_580r.pth",
        ),
    ],
}


def main():
    dlutils = load_dlutils()
    if dlutils is None:
        print("dlutils is unavailable or incompatible; falling back to S3 direct downloads.")

    for dataset, files in WEIGHTS.items():
        target_dir = os.path.join(ALAE_ROOT, "training_artifacts", dataset)
        for google_drive_id, s3_url in files:
            download_weight(dlutils, google_drive_id, s3_url, target_dir)


def load_dlutils():
    try:
        import dlutils
    except Exception as exc:
        print(f"Could not import dlutils: {exc}", file=sys.stderr)
        return None

    try:
        from packaging import version

        if not hasattr(dlutils, "__version__") or version.parse(dlutils.__version__) < version.parse("0.0.11"):
            print("dlutils is older than 0.0.11.", file=sys.stderr)
            return None
    except Exception as exc:
        print(f"Could not validate dlutils version: {exc}", file=sys.stderr)
        return None

    return dlutils


def download_weight(dlutils, google_drive_id, s3_url, target_dir):
    filename = os.path.basename(s3_url)
    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        print(f"Skipping existing file: {target_path}")
        return

    if dlutils is not None:
        try:
            print(f"Downloading {filename} from Google Drive...")
            dlutils.download.from_google_drive(google_drive_id, directory=target_dir)
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                return
            print(f"Google Drive download finished but {filename} was not found; trying S3.")
        except Exception as exc:
            print(f"Google Drive download failed for {filename}: {exc}")

    print(f"Downloading {filename} from S3...")
    download_from_url(s3_url, target_path)


def download_from_url(url, target_path):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    tmp_path = target_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    urllib.request.urlretrieve(url, tmp_path, reporthook=progress_hook(os.path.basename(target_path)))
    os.replace(tmp_path, target_path)
    print(f"\nSaved {target_path}")


def progress_hook(filename):
    def _hook(block_count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        percent = downloaded * 100.0 / total_size
        print(f"\r{filename}: {percent:5.1f}%", end="")

    return _hook


if __name__ == "__main__":
    main()
