from __future__ import annotations

from pathlib import Path
import urllib.request
import zipfile
import tarfile


DATASET_URLS = {
    # Public mirrors; users can override by setting paths.raw_csv directly.
    "nslkdd": {
        "url": "https://github.com/defcom17/NSL_KDD/archive/refs/heads/master.zip",
        "archive": "nslkdd_master.zip",
        "expected_csv": "NSL_KDD-master/KDDTrain+.txt",
    },
    "cicids2017": {
        "url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "archive": None,
        "expected_csv": None,
    },
    "cicddos2019": {
        "url": "https://www.unb.ca/cic/datasets/ddos-2019.html",
        "archive": None,
        "expected_csv": None,
    },
}


def _download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out_path)


def _extract(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(target_dir)
    elif archive_path.suffix in {".gz", ".tgz", ".tar"}:
        with tarfile.open(archive_path, "r:*" ) as tf:
            tf.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")


def ensure_dataset(cfg: dict) -> str:
    """Return a resolved local raw file path.

    Behavior:
    - If paths.raw_csv exists locally, use it.
    - If dataset.auto_download is false, fail.
    - For NSL-KDD: auto-download + extract and point to KDDTrain+.txt.
    - For CICIDS/CICDDoS: provide explicit actionable error because official sources are gated/HTML pages.
    """
    raw_csv = Path(cfg["paths"]["raw_csv"]) if cfg["paths"].get("raw_csv") else None
    if raw_csv and raw_csv.exists():
        return str(raw_csv)

    if not cfg.get("dataset", {}).get("auto_download", False):
        raise FileNotFoundError(
            f"Dataset file not found at {raw_csv}. Set dataset.auto_download=true or provide valid paths.raw_csv"
        )

    dname = cfg["dataset"]["name"].lower()
    if dname not in DATASET_URLS:
        raise ValueError(f"No downloader metadata for dataset: {dname}")

    entry = DATASET_URLS[dname]
    data_root = Path(cfg["paths"].get("data_root", "data")) / dname
    data_root.mkdir(parents=True, exist_ok=True)

    if dname == "nslkdd":
        archive = data_root / entry["archive"]
        if not archive.exists():
            _download(entry["url"], archive)
        extracted = data_root / "extracted"
        expected = extracted / entry["expected_csv"]
        if not expected.exists():
            _extract(archive, extracted)
        if not expected.exists():
            raise FileNotFoundError(f"Downloaded NSL-KDD but expected file missing: {expected}")
        return str(expected)

    raise RuntimeError(
        "Automatic download for CICIDS2017/CICDDoS2019 is not fully automatable from official pages in this script. "
        "Please place CSV exports locally and set paths.raw_csv."
    )
