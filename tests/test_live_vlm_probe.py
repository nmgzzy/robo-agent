"""真实 VLM 探针自身的离线回归；不访问远程模型。"""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from scripts.probe_live_vlm import _accuracy_checks, _make_probe_image


def test_probe_image_has_stable_geometry_and_colors():
    image_bytes = _make_probe_image()
    with Image.open(BytesIO(image_bytes)) as image:
        assert image.size == (320, 200)
        assert image.getpixel((0, 0)) == (255, 255, 255)
        assert image.getpixel((80, 90)) == (255, 0, 0)
        assert image.getpixel((240, 90)) == (0, 0, 255)


def test_accuracy_checks_accept_expected_chinese_observation():
    checks = _accuracy_checks("背景是白色，有2个大圆形，分别是红色和蓝色。")
    assert checks and all(checks.values())


def test_accuracy_checks_reject_incomplete_observation():
    checks = _accuracy_checks("背景是白色，有一个红色圆形。")
    assert checks["white_background"] is True
    assert checks["red_circle"] is True
    assert checks["two_circles"] is False
    assert checks["blue_circle"] is False
