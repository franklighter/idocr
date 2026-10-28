"""文本框 -> 行 -> 锚点字段。

为什么用「锚点 + 正则」而不是「像素模板切图」：
  * 身份证版式由国标固定，但拍照的边距、缩放、透视每次都不同，硬编码像素
    坐标非常脆；
  * 锚点（"姓名""公民身份号码"…）是语义不变量，OCR 读出来的位置天然跟着
    图像走，换版式也能用；
  * 正则跑在「按行拼接的全文」上，对 OCR 把标签和值拆成两个框、或合并成
    一个框的情况都成立。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .ocr import Token

# 字段名（内部 key，也是 CSV 列名）
F_NAME = "name"
F_GENDER = "gender"
F_ETHNICITY = "ethnicity"
F_BIRTH = "birth_date"
F_ADDRESS = "address"
F_ID = "id_number"
F_AUTHORITY = "issuing_authority"
F_PERIOD = "valid_period"

FRONT_FIELDS = (F_NAME, F_GENDER, F_ETHNICITY, F_BIRTH, F_ADDRESS, F_ID)
BACK_FIELDS = (F_AUTHORITY, F_PERIOD)
ALL_FIELDS = FRONT_FIELDS + BACK_FIELDS

# 出现在值后面的下一个标签，用来截断贪婪匹配
_STOP_LABELS = (
    "姓名",
    "性别",
    "民族",
    "出生",
    "住址",
    "公民身份号码",
    "公民身份证号",
    "身份号码",
    "身份证号码",
    "身份证号",
    "签发机关",
    "有效期限",
)


def normalize_text(text: str) -> str:
    """NFKC 归一（全角数字/字母/冒号 -> 半角）+ 去空白。"""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", text)


def group_lines(tokens: Sequence[Token], tol_ratio: float = 0.6) -> List[List[Token]]:
    """按纵向重叠把 token 聚成文本行，行内按 x 排序。"""
    if not tokens:
        return []

    heights = [t.height for t in tokens if t.height > 0]
    median_h = float(np.median(heights)) if heights else 12.0
    tol = max(6.0, median_h * tol_ratio)

    lines: List[Dict] = []
    for tok in sorted(tokens, key=lambda t: t.y_center):
        for line in lines:
            if abs(line["y"] - tok.y_center) <= tol:
                line["tokens"].append(tok)
                line["y"] = float(np.mean([t.y_center for t in line["tokens"]]))
                break
        else:
            lines.append({"y": tok.y_center, "tokens": [tok]})

    for line in lines:
        line["tokens"].sort(key=lambda t: t.x_left)
    lines.sort(key=lambda l: l["y"])
    return [line["tokens"] for line in lines]


def assemble(tokens: Sequence[Token]) -> str:
    """把 token 拼成「行内无分隔、行间换行」的全文，供正则使用。"""
    lines = group_lines(tokens)
    return "\n".join("".join(normalize_text(t.text) for t in line) for line in lines)


def _cut_at_label(value: str) -> str:
    """截掉值里混进来的下一个标签。"""
    cut = len(value)
    for label in _STOP_LABELS:
        idx = value.find(label)
        if 0 <= idx < cut:
            cut = idx
    return value[:cut].strip()


def _clean_punct(value: str) -> str:
    return value.strip(" 　:：,，.。;；-_/\\")


# ----------------------------------------------------------------------
# 人像面
# ----------------------------------------------------------------------
def parse_name(text: str) -> str:
    match = re.search(r"姓\s*名[：:]?([\u4e00-\u9fa5·]{2,10})", text)
    if not match:
        return ""
    name = _cut_at_label(match.group(1))
    name = re.sub(r"[·]{2,}", "·", name)
    return name if 2 <= len(name) <= 6 else ""


def parse_gender(text: str) -> str:
    match = re.search(r"性\s*别[：:]?([男女])", text)
    return match.group(1) if match else ""


def parse_ethnicity(text: str) -> str:
    match = re.search(r"民\s*族[：:]?([\u4e00-\u9fa5]{1,4})", text)
    if not match:
        return ""
    value = _cut_at_label(match.group(1))
    return value[:4]


def parse_birth(text: str) -> str:
    match = re.search(
        r"出\s*生[：:]?(\d{4})[年\.\-/](\d{1,2})[月\.\-/](\d{1,2})", text
    )
    if not match:
        # 有些拍摄会把「年月日」吃掉，只剩 19900101 这种形态
        match2 = re.search(r"出\s*生[：:]?(\d{4})(\d{2})(\d{2})", text)
        if not match2:
            return ""
        y, m, d = match2.groups()
    else:
        y, m, d = match.groups()
    month, day = int(m), int(d)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""
    return f"{int(y):04d}-{month:02d}-{day:02d}"


def parse_address(text: str) -> str:
    idx = text.find("住址")
    if idx < 0:
        return ""
    tail = text[idx + 2 :]
    # 去掉紧随其后的冒号
    tail = tail.lstrip(":：")
    cut = len(tail)
    for label in ("公民身份号码", "公民身份证号", "身份号码", "身份证号码", "身份证号", "签发机关", "有效期限"):
        pos = tail.find(label)
        if 0 <= pos < cut:
            cut = pos
    value = tail[:cut]
    value = re.sub(r"[\s　]+", "", value)
    value = _clean_punct(value)
    return value[:60]


_ID_LABEL_RE = re.compile(
    r"(?:公民身份号码|公民身份证号|身份号码|身份证号码|身份证号)[：:]?"
)
_DIGITISH_RE = re.compile(r"[0-9OoIlZzSsBbGg]{15,18}")

_LETTER_TO_DIGIT = {
    "O": "0",
    "o": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "i": "1",
    "l": "1",
    "L": "1",
    "Z": "2",
    "z": "2",
    "S": "5",
    "s": "5",
    "B": "8",
    "b": "8",
    "G": "6",
    "g": "9",
    "T": "7",
}


def sanitize_id(raw: str) -> str:
    """把 OCR 常见的字母/数字混淆还原成 18 位号码形态。"""
    out = []
    for idx, ch in enumerate(raw):
        if ch.isdigit():
            out.append(ch)
        elif ch in "Xx" and idx >= 15:  # X 只可能出现在末位
            out.append("X")
        elif ch in "Xx":
            out.append("0")
        else:
            out.append(_LETTER_TO_DIGIT.get(ch, ""))
    value = "".join(out)
    return value[:18]


def find_id_candidates(text: str) -> List[str]:
    """按可信度从高到低返回候选号码（去重保序）。"""
    candidates: List[str] = []

    # 1) 锚点后面的号码最可信
    for match in _ID_LABEL_RE.finditer(text):
        tail = text[match.end() : match.end() + 60]
        run = re.match(r"[^0-9OoIlZzSsBbGgXx]*([0-9OoIlZzSsBbGgXx]{15,18})", tail)
        if run:
            candidates.append(sanitize_id(run.group(1)))

    # 2) 全文里的长数字串兜底（防止标签被识别错）
    for match in _DIGITISH_RE.finditer(text):
        candidates.append(sanitize_id(match.group(0)))

    seen = set()
    unique: List[str] = []
    for cand in candidates:
        if len(cand) == 18 and cand not in seen:
            seen.add(cand)
            unique.append(cand)
    return unique


# ----------------------------------------------------------------------
# 国徽面
# ----------------------------------------------------------------------
def parse_authority(text: str) -> str:
    match = re.search(r"签发机关[：:]?(.{2,40}?)(?=有效期限|$)", text, flags=re.S)
    if not match:
        return ""
    value = re.sub(r"[\s　]+", "", match.group(1))
    value = _cut_at_label(value)
    return _clean_punct(value)[:30]


_DATE_RE = re.compile(r"(\d{4})[年\.\-/](\d{1,2})[月\.\-/](\d{1,2})")


def parse_valid_period(text: str) -> str:
    match = re.search(r"有效期限[：:]?(.{4,60})", text, flags=re.S)
    if not match:
        # 没有锚点就不要瞎猜：人像面上也可能出现日期（出生），误判代价很高
        return ""
    segment = re.sub(r"[\s　]+", "", match.group(1))

    dates = _DATE_RE.findall(segment)
    has_long_term = "长期" in segment

    if len(dates) >= 2:
        start = "%04d.%02d.%02d" % (int(dates[0][0]), int(dates[0][1]), int(dates[0][2]))
        end = "%04d.%02d.%02d" % (int(dates[1][0]), int(dates[1][1]), int(dates[1][2]))
        return f"{start}-{end}"
    if len(dates) == 1 and has_long_term:
        start = "%04d.%02d.%02d" % (int(dates[0][0]), int(dates[0][1]), int(dates[0][2]))
        return f"{start}-长期"
    if has_long_term:
        return "长期"
    return ""


# ----------------------------------------------------------------------
@dataclass
class Extraction:
    text: str = ""
    fields: Dict[str, str] = field(default_factory=dict)
    id_candidates: List[str] = field(default_factory=list)

    @property
    def side(self) -> str:
        has_front = any(self.fields.get(k) for k in FRONT_FIELDS)
        has_back = any(self.fields.get(k) for k in BACK_FIELDS)
        if has_front and has_back:
            return "both"
        if has_front:
            return "front"
        if has_back:
            return "back"
        return "unknown"


def extract(tokens: Sequence[Token]) -> Extraction:
    """从一次 OCR 结果中抽取所有字段（双面都试，字段互不冲突）。"""
    text = assemble(tokens)
    fields = {
        F_NAME: parse_name(text),
        F_GENDER: parse_gender(text),
        F_ETHNICITY: parse_ethnicity(text),
        F_BIRTH: parse_birth(text),
        F_ADDRESS: parse_address(text),
        F_AUTHORITY: parse_authority(text),
        F_PERIOD: parse_valid_period(text),
    }
    candidates = find_id_candidates(text)
    if candidates:
        fields[F_ID] = candidates[0]
    return Extraction(text=text, fields=fields, id_candidates=candidates)


def anchor_hits(text: str) -> int:
    return sum(1 for label in _STOP_LABELS if label in text)


def side_from_text(text: str) -> str:
    front = sum(k in text for k in ("姓名", "性别", "民族", "出生", "住址", "公民身份号码"))
    back = sum(k in text for k in ("签发机关", "有效期限"))
    return "front" if front >= back else "back"
