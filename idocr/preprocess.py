"""图像读写与多预处理变体生成。

思路：单次 OCR 在糊图/反光图上错一位的概率不低；但同一张图经过不同预处理
后，错误往往落在不同位置。把 N 个变体的结果做加权投票，再用校验位过滤，
可以把「随机错一位」压下去，而不需要换模型。
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import cv2
import numpy as np

# 变体默认权重：越能保留原始笔画信息的权重越高
VARIANT_WEIGHTS: Dict[str, float] = {
    "orig": 1.00,
    "gray": 0.90,
    "clahe": 1.00,
    "sharpen": 0.90,
    "upscale": 0.85,
    "binarize": 0.70,
    "denoise": 0.80,
}

DEFAULT_VARIANTS: tuple[str, ...] = ("orig", "clahe", "sharpen", "binarize")


def imread_any(path: str) -> np.ndarray:
    """读取图像，兼容 Windows 中文路径（cv2.imread 不支持非 ASCII 路径）。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise IOError(f"空文件或无法读取: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"无法解码为图像: {path}")
    return img


def imwrite_any(path: str, img: np.ndarray) -> None:
    """写入图像，兼容非 ASCII 路径。"""
    ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"编码失败: {path}")
    buf.tofile(path)


# ----------------------------------------------------------------------
# 单个变体
# ----------------------------------------------------------------------
def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _sharpen(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 3.0)
    return cv2.addWeighted(img, 1.6, blur, -0.6, 0)


def _upscale(img: np.ndarray, target: int = 1400) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) >= target:
        return img
    scale = min(2.0, target / max(h, w))
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _binarize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    out = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )
    return _to_bgr(out)


def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)


_BUILDERS = {
    "orig": lambda img: img,
    "gray": lambda img: _to_bgr(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
    "clahe": _clahe,
    "sharpen": _sharpen,
    "upscale": _upscale,
    "binarize": _binarize,
    "denoise": _denoise,
}


def build_variants(
    img: np.ndarray, names: Iterable[str] = DEFAULT_VARIANTS
) -> Dict[str, np.ndarray]:
    """按名称生成本预处理变体；未知名称会被忽略并保留在返回值之外。"""
    out: Dict[str, np.ndarray] = {}
    for name in names:
        name = name.strip().lower()
        builder = _BUILDERS.get(name)
        if builder is None:
            continue
        try:
            out[name] = builder(img)
        except cv2.error:  # 某变体失败不应中断整体流程
            continue
    if not out:
        out["orig"] = img
    return out


def variant_weight(name: str) -> float:
    return VARIANT_WEIGHTS.get(name, 0.8)


def list_variants() -> List[str]:
    return list(_BUILDERS)
