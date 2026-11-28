#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None


# =========================
# 颜色
# =========================

MIN_COLOR_AREA = 500

COLOR_RANGES = {
    "red": ((0, 0, 150), (90, 90, 255)),
    "green": ((0, 150, 0), (90, 255, 90)),
    "blue": ((150, 0, 0), (255, 90, 90)),
}


def color_mask(image, lower, upper):
    mask = cv2.inRange(
        image,
        np.array(lower, dtype=np.uint8),
        np.array(upper, dtype=np.uint8),
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )
    return mask


def find_blobs(mask, min_area=MIN_COLOR_AREA):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    blobs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        blobs.append(
            (
                int(cx),
                int(cy),
                int(radius),
                float(area),
            )
        )

    return blobs


def recognize_color(image):
    height, width = image.shape[:2]
    cx, cy = width // 2, height // 2

    # 使用 11×11 的中位数，减少边界混合和单像素噪声
    half = 5
    patch = image[
        cy - half:cy + half + 1,
        cx - half:cx + half + 1,
    ]

    b, g, r = np.median(
        patch.reshape(-1, 3),
        axis=0,
    )

    # 不要求亮度一定大于 150，而是判断哪个通道明显占优
    if r > 80 and r > 1.5 * g and r > 1.5 * b:
        return "red"

    if g > 80 and g > 1.5 * r and g > 1.5 * b:
        return "green"

    if b > 80 and b > 1.5 * r and b > 1.5 * g:
        return "blue"

    return None

# =========================
# 二维码
# =========================

QR_DETECTOR = cv2.QRCodeDetector()


def _qr_points(points):
    if points is None:
        return None

    points = np.asarray(points).reshape(-1, 2)
    if len(points) < 4:
        return None

    return points[:4]


def decode_qr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = []

    # 只调用一次多码接口
    ok, texts, points, _ = QR_DETECTOR.detectAndDecodeMulti(gray)

    if ok and texts and points is not None:
        for index, text in enumerate(texts):
            text = (text or "").strip()

            if not text:
                continue

            if index >= len(points):
                continue

            qr_points = np.asarray(points[index]).reshape(4, 2)

            results.append(
                {
                    "text": text,
                    "points": qr_points,
                }
            )

    if not results:
        text, single_points, _ = QR_DETECTOR.detectAndDecode(gray)
        text = (text or "").strip()

        if text:
            results.append(
                {
                    "text": text,
                    "points": _qr_points(single_points),
                }
            )

    # 必须返回结果
    return results


def recognize_qr(image):
    results = decode_qr(image)
    return results[0]["text"] if results else None


# =========================
# AprilTag
# =========================

FAMILIES = {
    "16h5": cv2.aruco.DICT_APRILTAG_16H5,
    "25h9": cv2.aruco.DICT_APRILTAG_25H9,
    "36h11": cv2.aruco.DICT_APRILTAG_36H11,
}

_TAG_DETECTORS = {}


def make_detector(family="36h11"):
    if family in _TAG_DETECTORS:
        return _TAG_DETECTORS[family]

    if family not in FAMILIES:
        raise ValueError("不支持的 AprilTag 家族：" + family)

    dictionary = cv2.aruco.getPredefinedDictionary(
        FAMILIES[family]
    )
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(
        dictionary,
        params,
    )

    result = (dictionary, detector)
    _TAG_DETECTORS[family] = result
    return result


def detect_tags(image, dictionary, detector):
    results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return results

    for index, tag_id in enumerate(ids.flatten()):
        points = np.asarray(
            corners[index],
            dtype=np.float32,
        ).reshape(4, 2)

        cx, cy = np.mean(points, axis=0)

        results.append(
            {
                "id": int(tag_id),
                "center": (float(cx), float(cy)),
                "corners": points,
            }
        )

    return results


def recognize_apriltag(image, family="36h11"):
    dictionary, detector = make_detector(family)
    results = detect_tags(
        image,
        dictionary,
        detector,
    )
    return str(results[0]["id"]) if results else None


# =========================
# OCR
# =========================

PSM = 7
LANG = "eng"
SCALE = 3.0
OCR_DEBUG_PATH = "/tmp/task4_ocr_processed.png"


def preprocess(gray):
    resized = cv2.resize(
        gray,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_CUBIC,
    )

    blurred = cv2.GaussianBlur(
        resized,
        (3, 3),
        0,
    )

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def read_text(image):
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is not installed"
        )

    config = "--psm %d" % PSM
    return pytesseract.image_to_string(
        image,
        lang=LANG,
        config=config,
    )


def recognize_ocr(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image

    height, width = gray.shape

    roi = gray[
        int(0.10 * height):int(0.90 * height),
        int(0.05 * width):int(0.95 * width),
    ]

    processed = preprocess(roi)
    cv2.imwrite(OCR_DEBUG_PATH, processed)

    raw = read_text(processed)
    text = re.sub(
        r"[^A-Z]",
        "",
        raw.upper(),
    )

    # 不在 vision_core 中依赖 rospy
    return text if text else None


# =========================
# YOLO
# =========================

_MODEL_CACHE = {}


def load_model(weights):
    if weights in _MODEL_CACHE:
        return _MODEL_CACHE[weights]

    from ultralytics import YOLO

    if os.path.sep in weights and not os.path.exists(weights):
        raise FileNotFoundError(
            "找不到 YOLO 权重：" + weights
        )

    model = YOLO(weights)
    _MODEL_CACHE[weights] = model
    return model


def detect_objects(
    model,
    image,
    conf=0.25,
    iou=0.45,
):
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    if not results:
        return []

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return []

    height, width = image.shape[:2]

    xyxy_values = boxes.xyxy.cpu().numpy()
    class_values = boxes.cls.cpu().numpy()
    confidence_values = boxes.conf.cpu().numpy()

    detections = []

    for xyxy, class_id, confidence in zip(
        xyxy_values,
        class_values,
        confidence_values,
    ):
        x1, y1, x2, y2 = [
            float(value) for value in xyxy
        ]

        detections.append(
            {
                "cls_id": int(class_id),
                "cls_name": result.names[int(class_id)],
                "conf": float(confidence),
                "xyxy": (x1, y1, x2, y2),
                "center": (
                    (x1 + x2) / (2.0 * width),
                    (y1 + y2) / (2.0 * height),
                ),
            }
        )

    return detections


def recognize_logo(
    image,
    weights,
    conf=0.5,
    iou=0.45,
):
    model = load_model(weights)
    detections = detect_objects(
        model,
        image,
        conf=conf,
        iou=iou,
    )

    if not detections:
        return None

    best = max(
        detections,
        key=lambda item: item["conf"],
    )
    return str(best["cls_name"])
