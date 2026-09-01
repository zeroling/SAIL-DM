"""Download locally or validate the bundled MedMNIST datasets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_URL = "https://zenodo.org/records/10519652/files"
SPECS = {
    "pathmnist": (
        "PathMNIST", "pathmnist.npz", (89996, 10004, 7180),
        "a8b06965200029087d5bd730944a56c1",
    ),
    "pathmnist224": (
        "PathMNIST224", "pathmnist_224.npz", (89996, 10004, 7180),
        "2c51a510bcdc9cf8ddb2af93af1eadec",
    ),
    "bloodmnist": (
        "BloodMNIST", "bloodmnist.npz", (11959, 1712, 3421),
        "7053d0359d879ad8a5505303e11de1dc",
    ),
    "dermamnist": (
        "DermaMNIST", "dermamnist.npz", (7007, 1003, 2005),
        "0744692d530f8e62ec473284d019b0c7",
    ),
    "organamnist": (
        "OrganAMNIST", "organamnist.npz", (34561, 6491, 17778),
        "68e3f8846a6bd62f0c9bf841c0d9eacc",
    ),
}


def validate(
    path: Path, counts: tuple[int, int, int], expected_md5: str
) -> None:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_md5 = digest.hexdigest()
    if actual_md5 != expected_md5:
        raise ValueError(
            f"{path.name} MD5 错误：{actual_md5} != {expected_md5}"
        )
    with np.load(path, allow_pickle=False) as payload:
        for split, expected in zip(("train", "val", "test"), counts):
            image_key = f"{split}_images"
            label_key = f"{split}_labels"
            if image_key not in payload or label_key not in payload:
                raise ValueError(f"{path.name} 缺少 {split} images/labels")
            image_count = int(payload[image_key].shape[0])
            label_count = int(payload[label_key].shape[0])
            if image_count != expected or label_count != expected:
                raise ValueError(
                    f"{path.name} {split} 数量错误："
                    f"{image_count}/{label_count} != {expected}"
                )


def download(name: str) -> Path:
    directory_name, filename, counts, expected_md5 = SPECS[name]
    destination = PROJECT_ROOT / "data" / directory_name / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        validate(destination, counts, expected_md5)
        print(f"已存在并通过校验：{destination}", flush=True)
        return destination

    temporary = destination.with_suffix(destination.suffix + ".download")
    url = f"{BASE_URL}/{filename}?download=1"
    print(f"正在下载 {name}：{url}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
        validate(temporary, counts, expected_md5)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"下载并校验完成：{destination}", flush=True)
    return destination


def check(name: str) -> Path:
    directory_name, filename, counts, expected_md5 = SPECS[name]
    destination = PROJECT_ROOT / "data" / directory_name / filename
    if not destination.is_file():
        raise FileNotFoundError(
            f"缺少本地数据文件：{destination}\n"
            "请重新上传包含 data/ 目录的离线压缩包。"
        )
    validate(destination, counts, expected_md5)
    print(f"本地文件通过校验：{destination}", flush=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download MedMNIST NPZ files into this project"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="默认下载全部四个数据集",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只校验包内已有文件，绝不访问网络",
    )
    args = parser.parse_args()
    selected = args.datasets or list(SPECS)
    unknown = [name for name in selected if name not in SPECS]
    if unknown:
        parser.error(
            f"未知数据集 {unknown}；可用：{', '.join(sorted(SPECS))}"
        )
    try:
        for name in selected:
            (check if args.check_only else download)(name)
    except Exception as error:
        action = "校验" if args.check_only else "下载"
        print(f"{action}失败：{error}", file=sys.stderr)
        return 1
    print("所选数据集全部通过校验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
