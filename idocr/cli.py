"""命令行入口。

用法::

    python run.py 图片或目录 -o results.csv
    python run.py samples --debug-dir debug --variants orig,clahe,sharpen,binarize
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Sequence

from .csvout import COLUMNS, append_rows
from .ocr import OcrEngine
from .pipeline import Options, process_image
from .preprocess import DEFAULT_VARIANTS, list_variants

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(paths: Sequence[str]) -> List[str]:
    files: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for root, _dirs, names in os.walk(path):
                for name in sorted(names):
                    if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                        files.append(os.path.join(root, name))
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise SystemExit(f"路径不存在: {path}")
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idocr",
        description="身份证识别：RapidOCR + 锚点行 + MOD 11-2 多预处理投票（纯 CPU）",
    )
    parser.add_argument("paths", nargs="+", help="图片文件或目录（可多个）")
    parser.add_argument("-o", "--csv", default="results.csv", help="结果 CSV（追加写）")
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help=f"预处理变体，逗号分隔。可选: {','.join(list_variants())}",
    )
    parser.add_argument("--debug-dir", default=None, help="保存矫正图/变体/全文，便于调参")
    parser.add_argument("--jsonl", default=None, help="同时把明细写成 JSONL")
    parser.add_argument("--no-detect", action="store_true", help="跳过卡片检测与透视矫正")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="低于该置信度标记待复核")
    parser.add_argument("--quiet", action="store_true", help="只在末尾打印汇总")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    images = _collect_images(args.paths)
    if not images:
        print("没有找到图片", file=sys.stderr)
        return 2

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    options = Options(
        variants=variants,
        detect_card=not args.no_detect,
        debug_dir=args.debug_dir,
        min_confidence=args.min_confidence,
    )

    engine = OcrEngine()
    backend = engine.warmup()
    rows = []
    print(f"共 {len(images)} 张图，后端 {backend}，变体 {variants}")

    for index, path in enumerate(images, 1):
        result = process_image(path, engine, options)
        row = result.as_row()
        rows.append(row)
        if not args.quiet:
            status = "REVIEW" if result.needs_review else "  OK  "
            print(
                f"[{index}/{len(images)}] {status} {os.path.basename(path)} "
                f"id={row['id_number'] or '-'} name={row['name'] or '-'} "
                f"({result.elapsed:.2f}s)"
            )

    append_rows(args.csv, rows)

    if args.jsonl:
        with open(args.jsonl, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    review = sum(1 for r in rows if r["needs_review"])
    checksum = sum(1 for r in rows if r["id_checksum_ok"])
    print(
        f"\n完成：{len(rows)} 张，校验通过 {checksum}，待复核 {review}；"
        f"结果已追加到 {os.path.abspath(args.csv)}"
    )
    print(f"列：{','.join(COLUMNS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
