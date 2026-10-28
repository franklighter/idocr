"""生成合成身份证样本（清晰 / 旋转 / 模糊 / 反光 / 摆拍透视），用于冒烟测试。

真实样本当然更好，但合成样本能覆盖"版式 + 退化"这两件事，适合回归。
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from idocr.validate import check_digit  # noqa: E402

W, H = 1012, 638
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

ID_PREFIX = "11010519900307123"
ID_NUMBER = ID_PREFIX + check_digit(ID_PREFIX)  # 110105199003071239

FRONT = {
    "name": "张伟",
    "gender": "男",
    "ethnicity": "汉",
    "birth": "1990年3月7日",
    "address1": "北京市朝阳区建国路88号院",
    "address2": "3号楼2单元601室",
    "id": ID_NUMBER,
}

BACK = {
    "authority": "北京市公安局朝阳分局",
    "period": "2015.07.01-2035.07.01",
}


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def _line(img: Image.Image, xy, text: str, size: int, fill=(40, 40, 40)) -> None:
    ImageDraw.Draw(img).text(xy, text, font=_font(size), fill=fill)


def render_front() -> Image.Image:
    img = Image.new("RGB", (W, H), (250, 249, 244))
    draw = ImageDraw.Draw(img)

    # 淡淡的防伪底纹
    for i in range(0, W, 26):
        draw.line([(i, 0), (i + 120, H)], fill=(238, 242, 236), width=1)

    _line(img, (330, 26), "中华人民共和国", 34, (170, 60, 60))
    _line(img, (392, 72), "居民身份证", 30, (170, 60, 60))

    # 头像占位
    draw.rectangle([800, 150, 960, 470], fill=(214, 216, 220), outline=(180, 182, 186))

    _line(img, (58, 150), "姓  名", 30)
    _line(img, (168, 150), FRONT["name"], 32)

    _line(img, (58, 212), "性  别", 30)
    _line(img, (132, 212), FRONT["gender"], 32)
    _line(img, (300, 212), "民  族", 30)
    _line(img, (374, 212), FRONT["ethnicity"], 32)

    _line(img, (58, 274), "出  生", 30)
    _line(img, (132, 274), FRONT["birth"], 30)

    _line(img, (58, 336), "住  址", 30)
    _line(img, (132, 336), FRONT["address1"], 28)
    _line(img, (132, 378), FRONT["address2"], 28)

    _line(img, (58, 470), "公民身份号码", 29)
    _line(img, (238, 468), FRONT["id"], 32)
    return img


def render_back() -> Image.Image:
    img = Image.new("RGB", (W, H), (250, 249, 244))
    draw = ImageDraw.Draw(img)
    for i in range(0, W, 26):
        draw.line([(i, 0), (i + 120, H)], fill=(238, 242, 236), width=1)

    _line(img, (300, 120), "中华人民共和国", 36, (170, 60, 60))
    _line(img, (360, 175), "居民身份证", 32, (170, 60, 60))

    _line(img, (70, 400), "签发机关", 30)
    _line(img, (206, 400), BACK["authority"], 28)

    _line(img, (70, 470), "有效期限", 30)
    _line(img, (206, 470), BACK["period"], 28)
    return img


def to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def degrade_rotate(bgr: np.ndarray, angle: float = 7.0) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(bgr, matrix, (W, H), borderValue=(252, 252, 250))


def degrade_blur(bgr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(bgr, (5, 5), 1.6)


def degrade_glare_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = gray.copy()
    cv2.ellipse(overlay, (640, 250), (330, 200), -20, 0, 360, (255, 255, 255), -1)
    blended = cv2.addWeighted(gray, 0.72, overlay, 0.45, 0)
    return cv2.GaussianBlur(blended, (3, 3), 1.0)


def degrade_on_desk(bgr: np.ndarray) -> np.ndarray:
    """把卡片贴到深色桌面并做透视变换，测试四边形检测。"""
    canvas_w, canvas_h = 1400, 1000
    rng = np.random.default_rng(7)
    desk = np.full((canvas_h, canvas_w, 3), 55, np.uint8)
    noise = rng.normal(0, 6, (canvas_h, canvas_w, 3)).astype(np.int16)
    desk = np.clip(desk.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = np.float32([[190, 130], [1170, 95], [1235, 855], [160, 900]])
    matrix = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(bgr, matrix, (canvas_w, canvas_h))
    mask = cv2.warpPerspective(
        np.full((H, W), 255, np.uint8), matrix, (canvas_w, canvas_h)
    )
    desk[mask > 0] = warped[mask > 0]
    return desk


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    front = to_bgr(render_front())
    back = to_bgr(render_back())

    samples = {
        "front_clear.png": front,
        "front_rotated.png": degrade_rotate(front, 6.5),
        "front_blur.png": degrade_blur(front),
        "front_glare_gray.png": degrade_glare_gray(front),
        "front_on_desk.png": degrade_on_desk(front),
        "back_clear.png": back,
        "back_rotated.png": degrade_rotate(back, -5.0),
    }

    for name, image in samples.items():
        path = os.path.join(OUT_DIR, name)
        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"编码失败: {name}")
        buf.tofile(path)
        print(f"wrote {path}")

    print(f"\n合成用身份证号: {ID_NUMBER}（校验位由 MOD 11-2 计算）")


if __name__ == "__main__":
    main()
