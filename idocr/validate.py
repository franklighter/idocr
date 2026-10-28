"""校验与纠错：GB 11643 的 MOD 11-2 校验位。

先说清楚校验位能干什么、不能干什么，这决定了代码怎么写：

**能**：任意单字符错误都会改变校验值（权重向量每个元素都与 11 互素），
所以「校验失败」是可靠的错误信号，1/11 的漏检率覆盖不了的是"两位同时错"。

**不能**：不要指望"逐位替换、直到校验通过"来纠错。第 i 位替换成某个数字后
校验通过的概率约 1/11，17 个位置扫一遍，几乎必然出现十几个"看似合法"的解，
完全无法判断哪个是真的。这类盲目搜索只会把错误号码变成"合法但错误"的号码，
比原来更危险。

因此本模块只做两件事：
1. **验证**（``checksum_ok``）——硬门槛；
2. **受约束的纠错**（``repair``）——只在"多预处理变体之间仅有个别位分歧"
   时，用候选字符集做替换搜索，且必须**恰好一个解**才采信；否则一律交给
   人工。
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, Iterable, List, Optional, Set, Tuple

WEIGHTS: Tuple[int, ...] = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
CHECK_CODES = "10X98765432"

# 行政区划省级前缀（前两位）
PROVINCE_PREFIXES: Set[str] = {
    "11", "12", "13", "14", "15",  # 华北
    "21", "22", "23",              # 东北
    "31", "32", "33", "34", "35", "36", "37",  # 华东
    "41", "42", "43", "44", "45", "46",        # 华中/华南
    "50", "51", "52", "53", "54",              # 西南
    "61", "62", "63", "64", "65",              # 西北
    "71", "81", "82", "91",                    # 台港澳及国外
}

ADDRESS_KEYWORDS = (
    "省", "市", "区", "县", "镇", "乡", "村", "街", "路", "巷", "号",
    "组", "栋", "幢", "单元", "室", "队", "委", "社区", "小区", "大道",
)


def normalize_id(value: str) -> str:
    if not value:
        return ""
    value = value.strip().replace(" ", "")
    if len(value) == 18:
        return value[:17] + value[17].upper()
    return value.upper()


def checksum_ok(value: str) -> bool:
    value = normalize_id(value)
    if len(value) != 18 or not value[:17].isdigit():
        return False
    total = sum(int(ch) * w for ch, w in zip(value[:17], WEIGHTS))
    return CHECK_CODES[total % 11] == value[17]


def check_digit(prefix: str) -> str:
    """给 17 位前缀算校验位（生成测试样本、自检用）。"""
    if len(prefix) != 17 or not prefix.isdigit():
        raise ValueError("需要 17 位数字前缀")
    total = sum(int(ch) * w for ch, w in zip(prefix, WEIGHTS))
    return CHECK_CODES[total % 11]


def repair(
    value: str, alternatives: Optional[Dict[int, Iterable[str]]] = None
) -> Optional[Tuple[str, bool]]:
    """受约束的单位纠错。

    ``alternatives`` 是"变体间出现过分歧的位置 -> 该位置出现过的字符集合"。
    只有在这种外部证据约束下做替换才有意义；不传 ``alternatives`` 时本函数
    不会做无约束搜索（那只会产生歧义解），只在号码本就合法时原样返回。

    返回 ``(号码, 是否改过)``；无法判定时返回 ``None``。
    """
    value = normalize_id(value)
    if len(value) != 18 or not value[:17].isdigit():
        return None
    if checksum_ok(value):
        return value, False
    if not alternatives:
        return None

    solutions: List[str] = []
    for pos, chars in alternatives.items():
        if not (0 <= pos < 18):
            continue
        for ch in set(chars):
            if ch == value[pos]:
                continue
            candidate = value[:pos] + ch + value[pos + 1 :]
            if checksum_ok(candidate):
                solutions.append(candidate)

    unique = list(dict.fromkeys(solutions))
    if len(unique) == 1:
        return unique[0], True
    return None


def id_birth(value: str) -> str:
    value = normalize_id(value)
    if len(value) != 18 or not value[6:14].isdigit():
        return ""
    try:
        date = _dt.date(int(value[6:10]), int(value[10:12]), int(value[12:14]))
    except ValueError:
        return ""
    return date.isoformat()


def id_gender(value: str) -> str:
    value = normalize_id(value)
    if len(value) != 18 or not value[16].isdigit():
        return ""
    return "男" if int(value[16]) % 2 == 1 else "女"


def id_region(value: str) -> str:
    value = normalize_id(value)
    return value[:6] if len(value) >= 6 else ""


def region_ok(value: str) -> bool:
    return id_region(value)[:2] in PROVINCE_PREFIXES


def address_ok(value: str) -> bool:
    return bool(value) and any(kw in value for kw in ADDRESS_KEYWORDS)


def cross_check(fields: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """用合法身份证号反推出生/性别/行政区划，与识别结果交叉验证。

    号码通过校验时，它派生出的信息比 OCR 单独读出的字段更可信，因此冲突时
    以号码为准，但会在 notes 里留痕，方便审计。
    """
    notes: List[str] = []
    number = normalize_id(fields.get("id_number", ""))
    if not checksum_ok(number):
        return fields, notes

    derived_birth = id_birth(number)
    if derived_birth:
        current = fields.get("birth_date", "")
        if current and current != derived_birth:
            notes.append(f"birth_mismatch:{current}->{derived_birth}")
        fields["birth_date"] = derived_birth

    derived_gender = id_gender(number)
    if derived_gender:
        current = fields.get("gender", "")
        if current and current != derived_gender:
            notes.append(f"gender_mismatch:{current}->{derived_gender}")
        fields["gender"] = derived_gender

    if not region_ok(number):
        notes.append("region_prefix_unknown")
    if fields.get("address") and not address_ok(fields["address"]):
        notes.append("address_keyword_missing")

    return fields, notes
