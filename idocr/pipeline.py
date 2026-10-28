"""编排：预处理变体 -> OCR -> 抽取 -> 加权投票 -> 校验 -> 结果。"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from . import validate as V
from .card import normalize_card
from .extract import (
    ALL_FIELDS,
    F_AUTHORITY,
    F_ID,
    F_NAME,
    F_PERIOD,
    Extraction,
    assemble,
    anchor_hits,
    extract,
)
from .ocr import OcrEngine, Token
from .preprocess import (
    DEFAULT_VARIANTS,
    build_variants,
    imread_any,
    imwrite_any,
    variant_weight,
)

logger = logging.getLogger(__name__)


@dataclass
class Options:
    variants: Sequence[str] = DEFAULT_VARIANTS
    detect_card: bool = True
    debug_dir: Optional[str] = None
    min_confidence: float = 0.5


@dataclass
class Result:
    image: str
    side: str = "unknown"
    fields: Dict[str, str] = field(default_factory=lambda: {k: "" for k in ALL_FIELDS})
    confidence: Dict[str, float] = field(default_factory=dict)
    id_checksum_ok: bool = False
    id_repaired: bool = False
    needs_review: bool = True
    notes: List[str] = field(default_factory=list)
    text: str = ""
    mode: str = "raw"
    elapsed: float = 0.0

    def as_row(self) -> Dict[str, object]:
        """拍平成 CSV 一行。"""
        row: Dict[str, object] = {
            "image": self.image,
            "side": self.side,
            "mode": self.mode,
        }
        row.update({k: self.fields.get(k, "") for k in ALL_FIELDS})
        row["id_checksum_ok"] = int(self.id_checksum_ok)
        row["id_repaired"] = int(self.id_repaired)
        row["conf_name"] = round(self.confidence.get(F_NAME, 0.0), 3)
        row["conf_id_number"] = round(self.confidence.get(F_ID, 0.0), 3)
        row["needs_review"] = int(self.needs_review)
        row["notes"] = ";".join(self.notes)
        row["elapsed_s"] = round(self.elapsed, 3)
        return row


# ----------------------------------------------------------------------
def _rotate_180(img):
    return cv2.rotate(img, cv2.ROTATE_180)


def _best_orientation(
    card, engine: OcrEngine
) -> Tuple[object, List[Token], str]:
    """正立 vs 180°：用锚点命中数投票。只在可疑时才多跑一次 OCR。"""
    tokens = engine.run(card)
    hits = anchor_hits(assemble(tokens))
    if hits >= 2:
        return card, tokens, ""

    rotated = _rotate_180(card)
    tokens_r = engine.run(rotated)
    if anchor_hits(assemble(tokens_r)) > hits:
        return rotated, tokens_r, "rotated180"
    return card, tokens, ""


def _weighted_field_vote(
    extractions: Sequence[Tuple[str, float, Extraction]]
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """非号码字段：加权多数投票。"""
    counters: Dict[str, Counter] = {}
    for _name, weight, ext in extractions:
        for key, value in ext.fields.items():
            if key == F_ID or not value:
                continue
            counters.setdefault(key, Counter())[value] += weight

    fields: Dict[str, str] = {}
    confidence: Dict[str, float] = {}
    for key, counter in counters.items():
        total = sum(counter.values())
        if total <= 0:
            continue
        value, win = counter.most_common(1)[0]
        fields[key] = value
        confidence[key] = win / total
    return fields, confidence


def _position_consensus(
    candidates: Sequence[Tuple[str, float]]
) -> Tuple[str, List[Counter], float]:
    """把多个 18 位候选按位对齐投票，得到共识序列与每位的一致性。"""
    votes = [Counter() for _ in range(18)]
    for cand, weight in candidates:
        if len(cand) != 18:
            continue
        for pos, ch in enumerate(cand):
            votes[pos][ch] += weight

    chars: List[str] = []
    confidences: List[float] = []
    for counter in votes:
        total = sum(counter.values())
        if total <= 0:
            chars.append("")
            confidences.append(0.0)
            continue
        ch, win = counter.most_common(1)[0]
        chars.append(ch)
        confidences.append(win / total)

    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return "".join(chars), votes, confidence


def _vote_id_number(
    extractions: Sequence[Tuple[str, float, Extraction]]
) -> Tuple[str, float, bool, bool, List[str]]:
    """号码单独处理：校验位优先，其次按位共识，最后做受约束纠错。

    顺序很重要——先看有没有合法解，才能避免"把错号码修成合法错号码"。
    返回 ``(号码, 置信度, 校验通过, 是否被修正, 备注)``。
    """
    notes: List[str] = []
    weights: Counter = Counter()
    for _name, weight, ext in extractions:
        for rank, cand in enumerate(ext.id_candidates[:3]):
            weights[cand] += weight * (1.0 if rank == 0 else 0.5)

    if not weights:
        return "", 0.0, False, False, ["id_not_found"]

    total = sum(weights.values())
    ordered = sorted(weights.items(), key=lambda kv: -kv[1])

    # 1) 有候选直接通过校验 -> 取权重最高的合法解
    valid = [(c, w) for c, w in ordered if V.checksum_ok(c)]
    if valid:
        best, win = valid[0]
        if len(valid) > 1:
            notes.append(f"id_multiple_valid:{len(valid)}")
            notes.append("id_ambiguous")
        return best, win / total, True, False, notes

    # 2) 按位共识
    consensus, votes, consensus_conf = _position_consensus(ordered)
    if V.checksum_ok(consensus):
        notes.append("id_from_consensus")
        return consensus, consensus_conf, True, True, notes

    # 3) 只在"变体间有分歧"的位置上做替换，且必须恰好一个解
    alternatives: Dict[int, List[str]] = {}
    for pos, counter in enumerate(votes):
        pos_total = sum(counter.values())
        if pos_total <= 0:
            continue
        ranked = counter.most_common()
        if ranked[0][1] / pos_total >= 0.75:
            continue  # 该位高度一致，不是可疑位
        alternatives[pos] = [ch for ch, _w in ranked[:3]]

    outcome = V.repair(consensus, alternatives) if alternatives else None
    if outcome is not None:
        fixed, changed = outcome
        notes.append(f"id_repaired:{consensus}->{fixed}" if changed else "id_consensus_fix")
        return fixed, consensus_conf, True, True, notes

    # 4) 无解 -> 交出最可能的原始候选，标记待复核
    notes.append("id_unrecoverable" if not alternatives else "id_repair_ambiguous")
    best, win = ordered[0]
    return best, win / total, False, False, notes


# ----------------------------------------------------------------------
def process_image(path: str, engine: OcrEngine, options: Optional[Options] = None) -> Result:
    options = options or Options()
    started = time.time()
    result = Result(image=path)

    try:
        image = imread_any(path)
    except Exception as exc:
        result.notes.append(f"read_error:{exc}")
        result.elapsed = time.time() - started
        return result

    card, mode = normalize_card(image, enable_detect=options.detect_card)
    result.mode = mode

    card, base_tokens, rotation_note = _best_orientation(card, engine)
    if rotation_note:
        result.notes.append(rotation_note)

    variants = build_variants(card, options.variants)
    extractions: List[Tuple[str, float, Extraction]] = []
    for name, variant in variants.items():
        tokens: List[Token] = base_tokens if name == "orig" else engine.run(variant)
        extractions.append((name, variant_weight(name), extract(tokens)))
        if options.debug_dir:
            _dump_debug(options.debug_dir, path, name, variant, extractions[-1])

    fields, confidence = _weighted_field_vote(extractions)
    provisional_side = _decide_side(fields)
    number, id_conf, id_ok, id_fixed, id_notes = _vote_id_number(extractions)

    # 国徽面本来就没有号码，不该报"未找到号码"
    if provisional_side == "back" and not number:
        id_notes = []

    result.notes.extend(id_notes)
    if number:
        fields[F_ID] = number
        confidence[F_ID] = id_conf
    result.id_checksum_ok = id_ok
    result.id_repaired = id_fixed

    fields, cross_notes = V.cross_check(fields)
    result.notes.extend(cross_notes)

    # 号码不合法时，出生/性别字段退回 OCR 原始投票结果，避免被污染
    result.fields.update({k: fields.get(k, "") for k in ALL_FIELDS})
    result.confidence = confidence
    result.text = extractions[0][2].text if extractions else ""

    result.side = _decide_side(result.fields)

    if result.side == "back":
        required = (F_AUTHORITY, F_PERIOD)
    else:
        required = (F_NAME, F_ID)
    weak = [k for k in required if not result.fields.get(k)]
    low_conf = [
        k
        for k, c in confidence.items()
        if c < options.min_confidence and result.fields.get(k)
    ]
    if low_conf:
        result.notes.append("low_confidence:" + ",".join(sorted(low_conf)))

    result.needs_review = bool(
        (result.side != "back" and not result.id_checksum_ok)
        or weak
        or result.side == "unknown"
        or low_conf
    )
    result.elapsed = time.time() - started
    return result


def _decide_side(fields: Dict[str, str]) -> str:
    front = any(fields.get(k) for k in ("name", "gender", "ethnicity", "birth_date", "address", "id_number"))
    back = any(fields.get(k) for k in ("issuing_authority", "valid_period"))
    if front and back:
        return "both"
    if front:
        return "front"
    if back:
        return "back"
    return "unknown"


def _dump_debug(debug_dir: str, image_path: str, variant: str, img, ext: Extraction) -> None:
    try:
        os.makedirs(debug_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        imwrite_any(os.path.join(debug_dir, f"{stem}__{variant}.png"), img)
        if variant == "orig":
            with open(
                os.path.join(debug_dir, f"{stem}__text.txt"), "w", encoding="utf-8"
            ) as handle:
                handle.write(ext.text)
    except Exception:  # 调试输出失败不应影响主流程
        logger.debug("debug dump failed", exc_info=True)
