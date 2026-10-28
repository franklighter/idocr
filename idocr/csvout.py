"""CSV 追加写出（一行一张图，UTF-8 BOM 便于 Excel 直接打开）。"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, Iterable, List

COLUMNS: List[str] = [
    "timestamp",
    "image",
    "side",
    "mode",
    "name",
    "gender",
    "ethnicity",
    "birth_date",
    "address",
    "id_number",
    "issuing_authority",
    "valid_period",
    "id_checksum_ok",
    "id_repaired",
    "conf_name",
    "conf_id_number",
    "needs_review",
    "notes",
    "elapsed_s",
]


def append_rows(path: str, rows: Iterable[Dict[str, object]]) -> int:
    """把结果追加到 CSV；文件不存在时自动写表头。返回写入行数。"""
    rows = list(rows)
    if not rows:
        return 0

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    # 注意：仅在新建文件时使用 utf-8-sig，避免每次追加都写入 BOM
    encoding = "utf-8-sig" if is_new else "utf-8"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        for row in rows:
            payload = {"timestamp": timestamp}
            payload.update(row)
            writer.writerow(payload)
    return len(rows)
