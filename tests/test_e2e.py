"""端到端冒烟测试：需要已安装 RapidOCR，单次约 15-25 秒。

如果没有样本图会先调用 make_sample 生成。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idocr.ocr import OcrEngine  # noqa: E402
from idocr.pipeline import Options, process_image  # noqa: E402
from idocr.validate import check_digit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")
EXPECTED_ID = "11010519900307123" + check_digit("11010519900307123")


def setUpModule() -> None:
    if not os.path.isdir(SAMPLES) or not os.listdir(SAMPLES):
        import make_sample  # type: ignore

        make_sample.main()


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = OcrEngine()
        cls.engine.warmup()
        cls.options = Options()

    def _run(self, name: str):
        return process_image(os.path.join(SAMPLES, name), self.engine, self.options)

    def test_front_clear(self):
        result = self._run("front_clear.png")
        self.assertEqual(result.fields["name"], "张伟")
        self.assertEqual(result.fields["id_number"], EXPECTED_ID)
        self.assertTrue(result.id_checksum_ok)
        self.assertEqual(result.side, "front")
        self.assertFalse(result.needs_review)

    def test_back_clear(self):
        result = self._run("back_clear.png")
        self.assertEqual(result.side, "back")
        self.assertIn("公安局", result.fields["issuing_authority"])
        self.assertTrue(result.fields["valid_period"].startswith("2015.07.01"))
        self.assertFalse(result.needs_review)

    def test_degraded_images(self):
        for name in ("front_blur.png", "front_rotated.png", "front_on_desk.png"):
            with self.subTest(image=name):
                result = self._run(name)
                self.assertEqual(result.fields["id_number"], EXPECTED_ID)
                self.assertTrue(result.id_checksum_ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
