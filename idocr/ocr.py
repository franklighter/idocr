"""RapidOCR 适配层。

同时兼容两个发行包，按可用性自动选择：

* ``rapidocr``             (>=3.0, 新版 API，返回 ``RapidOCROutput`` 对象)
* ``rapidocr_onnxruntime`` (1.x 经典 API，返回 ``(result, elapse)`` 元组)

两者都只在 CPU 上跑 ONNXRuntime，安装后即可离线使用（模型随 wheel 打包或
在首次运行时下载到本地缓存）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """一个文本框及其识别结果（坐标为原图像素，4x2）。"""

    text: str
    box: np.ndarray
    score: float

    # -- 便于排序/分组的几何属性 -------------------------------------------
    @property
    def x_left(self) -> float:
        return float(np.min(self.box[:, 0]))

    @property
    def x_center(self) -> float:
        return float(np.mean(self.box[:, 0]))

    @property
    def y_center(self) -> float:
        return float(np.mean(self.box[:, 1]))

    @property
    def height(self) -> float:
        return float(np.max(self.box[:, 1]) - np.min(self.box[:, 1]))


class OcrEngine:
    """线程不安全的 RapidOCR 包装。多进程请每个进程一个实例。"""

    def __init__(self) -> None:
        self._impl = None
        self.backend: Optional[str] = None

    # ------------------------------------------------------------------
    def _load(self):
        if self._impl is not None:
            return self._impl

        try:  # 新包
            from rapidocr import RapidOCR  # type: ignore

            self._impl = RapidOCR()
            self.backend = "rapidocr"
            logger.info("使用 rapidocr (>=3.0) 后端")
            return self._impl
        except Exception:  # pragma: no cover - 取决于安装环境
            pass

        try:  # 旧包
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            self._impl = RapidOCR()
            self.backend = "rapidocr_onnxruntime"
            logger.info("使用 rapidocr_onnxruntime (1.x) 后端")
            return self._impl
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "未找到 RapidOCR。请任选其一安装：\n"
                "  pip install rapidocr_onnxruntime   # 轻量、模型随包，推荐离线部署\n"
                "  pip install rapidocr onnxruntime   # 新版"
            ) from exc

    # ------------------------------------------------------------------
    def warmup(self) -> str:
        """提前加载模型（首次约 1-2 秒），返回实际使用的后端名。"""
        self._load()
        return self.backend or "unknown"

    # ------------------------------------------------------------------
    def run(self, image: np.ndarray) -> List[Token]:
        """对 BGR 图像做检测+识别，返回 Token 列表（按阅读顺序粗排）。"""
        engine = self._load()
        if image is None or image.size == 0:
            return []

        raw = engine(image)
        tokens = self._parse(raw)
        tokens.sort(key=lambda t: (round(t.y_center / 12.0), t.x_left))
        return tokens

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(raw) -> List[Token]:
        tokens: List[Token] = []

        # 新 API: RapidOCROutput(boxes=..., txts=..., scores=...)
        if hasattr(raw, "txts"):
            txts = getattr(raw, "txts", None)
            boxes = getattr(raw, "boxes", None)
            scores = getattr(raw, "scores", None)
            if txts is None or boxes is None:
                return []
            for text, box, score in zip(txts, boxes, scores or [1.0] * len(txts)):
                if text is None:
                    continue
                tokens.append(
                    Token(str(text), np.asarray(box, dtype=np.float32), float(score))
                )
            return tokens

        # 旧 API: (result, elapse)，result 为 [[box, text, score], ...] 或 None
        if isinstance(raw, tuple):
            result = raw[0]
        else:
            result = raw
        if not result:
            return []

        for item in result:
            try:
                box, text, score = item[0], item[1], item[2]
            except (TypeError, IndexError):  # 容错：个别版本结构略有差异
                continue
            if not text:
                continue
            tokens.append(
                Token(str(text), np.asarray(box, dtype=np.float32), float(score))
            )
        return tokens


def anchor_score(tokens: Sequence[Token]) -> float:
    """用于判断卡片是否被 180° 翻转：命中关键词越多、置信度越高越像正立。"""
    keywords = (
        "姓名",
        "性别",
        "民族",
        "出生",
        "住址",
        "公民身份号码",
        "签发机关",
        "有效期限",
        "居民身份证",
    )
    score = 0.0
    for tok in tokens:
        for kw in keywords:
            if kw in tok.text:
                score += 1.0 + tok.score
                break
    return score
