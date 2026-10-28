"""卡片检测、透视矫正与方向校正。

身份证是 85.6 x 54.0 mm 的标准矩形（长宽比 1.585），这个先验非常适合用
传统 CV 做定位——不需要额外模型，也就没有「模型能不能跑在平头哥上」的问题。

检测失败时不报错，退回原图继续走 OCR：锚点解析对轻微倾斜是容忍的，只是
精度会下降，属于可接受的降级。
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CARD_W, CARD_H = 1012, 638  # 85.6:54 ≈ 1.586
CARD_ASPECT = CARD_W / CARD_H


def _order_points(pts: np.ndarray) -> np.ndarray:
    """把任意顺序的四个角点整理成 左上、右上、右下、左下。"""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_score(quad: np.ndarray, area: float, img_area: float) -> float:
    """按 面积占比 x 矩形度 x 长宽比接近度 给候选打分。

    几个硬约束是踩过坑的：卡片是横向矩形（w/h≈1.585）。如果不限制方向，
    人像面上的头像框（竖向矩形，面积很小但长宽比也在范围内）会被误判成
    卡片，透视变换后整张图就只剩一个灰块。
    """
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if w < 10 or h < 10:
        return 0.0

    aspect = w / h  # 注意：不是 max/min，方向很重要
    if not (1.15 <= aspect <= 2.35):
        return 0.0

    fill = area / max(w * h, 1.0)
    if fill < 0.70:
        return 0.0

    area_ratio = area / max(img_area, 1.0)
    if area_ratio < 0.03:
        return 0.0

    aspect_penalty = max(0.0, 1.0 - abs(aspect - CARD_ASPECT) / CARD_ASPECT)
    # 面积开方：让"更大的四边形"优先，但不至于完全压倒长宽比
    return float((area_ratio**0.5) * fill * aspect_penalty)


def detect_card_quad(img: np.ndarray) -> Optional[np.ndarray]:
    """返回四个角点（原图坐标），找不到可信四边形时返回 None。"""
    h, w = img.shape[:2]
    if h < 60 or w < 60:
        return None

    scale = min(1.0, 1000.0 / max(h, w))
    small = (
        cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else img
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    masks = []

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    masks.append(edges)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    masks.append(otsu)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    _, sat = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    masks.append(sat)

    img_area = float(small.shape[0] * small.shape[1])
    best_score, best_quad, best_area = 0.0, None, 0.0

    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 0.05 * img_area:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                # 轮廓毛刺太多时退回最小外接旋转矩形
                box = cv2.boxPoints(cv2.minAreaRect(contour))
                approx = box.reshape(-1, 1, 2).astype(np.int32)
            quad = _order_points(np.asarray(approx, dtype=np.float32))
            score = _quad_score(quad, area, img_area)
            if score > best_score:
                best_score, best_quad, best_area = score, quad, area

    if best_quad is None or best_score < 0.45:
        return None
    # 卡在画面里时一定是个"主体"；面积太小的四边形基本都是头像框、印章、
    # 或者桌面上的其他矩形物体，宁可退回 resize 也不要错误地裁切。
    if best_area / img_area < 0.30:
        logger.debug("largest quad too small: %.3f", best_area / img_area)
        return None

    if scale < 1.0:
        best_quad = best_quad / scale
    return best_quad.astype(np.float32)


def warp_card(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """按四个角点做透视变换，输出固定尺寸的卡片图。"""
    dst = np.array(
        [[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(
        img, matrix, (CARD_W, CARD_H), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )


def normalize_card(img: np.ndarray, enable_detect: bool = True) -> Tuple[np.ndarray, str]:
    """检测 + 矫正；失败时降级。

    返回值第二项是处理方式，用于日志/CSV 备注：
    ``warp`` / ``resize`` / ``raw``。
    """
    if enable_detect:
        quad = detect_card_quad(img)
        if quad is not None:
            return warp_card(img, quad), "warp"

    h, w = img.shape[:2]
    aspect = max(h, w) / max(min(h, w), 1)
    if 1.25 <= aspect <= 2.0:
        # 已经是卡片比例，直接缩放到标准尺寸
        interp = cv2.INTER_AREA if w > CARD_W else cv2.INTER_CUBIC
        return cv2.resize(img, (CARD_W, CARD_H), interpolation=interp), "resize"

    return img, "raw"
