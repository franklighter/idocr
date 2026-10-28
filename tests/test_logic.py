"""纯逻辑单元测试（不需要 OCR 引擎 / 图像）。

覆盖：校验位、受约束纠错、锚点解析、交叉校验、CSV 追加。
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idocr import validate as V  # noqa: E402
from idocr.csvout import COLUMNS, append_rows  # noqa: E402
from idocr.extract import Token, assemble, extract  # noqa: E402

VALID_ID = "110105199003071239"


def make_tokens(lines):
    """lines: [[(text, x, y, w, h), ...], ...]"""
    tokens = []
    for line in lines:
        for text, x, y, w, h in line:
            box = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
            )
            tokens.append(Token(text=text, box=box, score=0.95))
    return tokens


FRONT_TOKENS = make_tokens(
    [
        [("姓名", 58, 150, 80, 34), ("张伟", 168, 150, 80, 34)],
        [("性别", 58, 212, 80, 34), ("男", 132, 212, 40, 34), ("民族", 300, 212, 80, 34), ("汉", 374, 212, 40, 34)],
        [("出生", 58, 274, 80, 34), ("1990年3月7日", 132, 274, 220, 34)],
        [("住址", 58, 336, 80, 34), ("北京市朝阳区建国路88号院", 132, 336, 300, 34)],
        [("3号楼2单元601室", 132, 378, 240, 34)],
        [("公民身份号码", 58, 470, 170, 34), (VALID_ID, 238, 468, 260, 34)],
    ]
)

BACK_TOKENS = make_tokens(
    [
        [("签发机关", 70, 400, 130, 34), ("北京市公安局朝阳分局", 206, 400, 330, 34)],
        [("有效期限", 70, 470, 130, 34), ("2015.07.01-2035.07.01", 206, 470, 330, 34)],
    ]
)


class ChecksumTests(unittest.TestCase):
    def test_valid_number(self):
        self.assertTrue(V.checksum_ok(VALID_ID))

    def test_invalid_number(self):
        broken = VALID_ID[:10] + ("0" if VALID_ID[10] != "0" else "1") + VALID_ID[11:]
        self.assertFalse(V.checksum_ok(broken))

    def test_check_digit_roundtrip(self):
        self.assertEqual(VALID_ID[-1], V.check_digit(VALID_ID[:17]))

    def test_birth_and_gender(self):
        self.assertEqual(V.id_birth(VALID_ID), "1990-03-07")
        self.assertEqual(V.id_gender(VALID_ID), "男")

    def test_region(self):
        self.assertTrue(V.region_ok(VALID_ID))
        self.assertFalse(V.region_ok("990105199003071239"))


class RepairTests(unittest.TestCase):
    def test_no_alternatives_returns_none(self):
        broken = VALID_ID[:5] + ("0" if VALID_ID[5] != "0" else "1") + VALID_ID[6:]
        self.assertIsNone(V.repair(broken))

    def test_constrained_repair_single_position(self):
        pos = 9
        wrong = "0" if VALID_ID[pos] != "0" else "1"
        broken = VALID_ID[:pos] + wrong + VALID_ID[pos + 1 :]
        fixed = V.repair(broken, {pos: [wrong, VALID_ID[pos]]})
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed[0], VALID_ID)
        self.assertTrue(fixed[1])

    def test_already_valid(self):
        outcome = V.repair(VALID_ID, {0: ["1"]})
        self.assertEqual(outcome, (VALID_ID, False))


class ExtractTests(unittest.TestCase):
    def test_front_fields(self):
        ext = extract(FRONT_TOKENS)
        self.assertEqual(ext.fields["name"], "张伟")
        self.assertEqual(ext.fields["gender"], "男")
        self.assertEqual(ext.fields["ethnicity"], "汉")
        self.assertEqual(ext.fields["birth_date"], "1990-03-07")
        self.assertEqual(
            ext.fields["address"], "北京市朝阳区建国路88号院3号楼2单元601室"
        )
        self.assertEqual(ext.fields["id_number"], VALID_ID)
        self.assertEqual(ext.side, "front")

    def test_back_fields(self):
        ext = extract(BACK_TOKENS)
        self.assertEqual(ext.fields["issuing_authority"], "北京市公安局朝阳分局")
        self.assertEqual(ext.fields["valid_period"], "2015.07.01-2035.07.01")
        self.assertEqual(ext.side, "back")

    def test_merged_label_and_value(self):
        tokens = make_tokens([[("姓名张伟性别男民族汉", 58, 150, 400, 34)]])
        ext = extract(tokens)
        self.assertEqual(ext.fields["name"], "张伟")
        self.assertEqual(ext.fields["gender"], "男")
        self.assertEqual(ext.fields["ethnicity"], "汉")

    def test_fullwidth_digits(self):
        tokens = make_tokens(
            [[("公民身份号码", 58, 470, 170, 34), ("１１０１０５１９９００３０７１２３９", 238, 470, 300, 34)]]
        )
        ext = extract(tokens)
        self.assertEqual(ext.fields["id_number"], VALID_ID)

    def test_assemble_reads_in_order(self):
        text = assemble(FRONT_TOKENS)
        self.assertTrue(text.startswith("姓名张伟"))
        self.assertIn("公民身份号码" + VALID_ID, text)


class CrossCheckTests(unittest.TestCase):
    def test_birth_override(self):
        fields = {"id_number": VALID_ID, "birth_date": "1990-03-08", "gender": "女"}
        fixed, notes = V.cross_check(dict(fields))
        self.assertEqual(fixed["birth_date"], "1990-03-07")
        self.assertEqual(fixed["gender"], "男")
        self.assertTrue(any("birth_mismatch" in n for n in notes))
        self.assertTrue(any("gender_mismatch" in n for n in notes))

    def test_invalid_id_leaves_fields(self):
        fields = {"id_number": "110105199003071238", "birth_date": "1990-03-07"}
        fixed, notes = V.cross_check(dict(fields))
        self.assertEqual(fixed["birth_date"], "1990-03-07")
        self.assertEqual(notes, [])


class CsvTests(unittest.TestCase):
    def test_append_creates_header_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            append_rows(path, [{"image": "a.png", "name": "张三"}])
            append_rows(path, [{"image": "b.png", "name": "李四"}])
            with open(path, encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows[0].keys()), COLUMNS)
            self.assertEqual(rows[1]["name"], "李四")


if __name__ == "__main__":
    unittest.main(verbosity=2)
