# YOLOv12 inference through ONNX Runtime instead of PyTorch.
#
# Why this module exists: the container only ever runs inference, and PyTorch
# costs 728 MB of the image for that privilege (measured with `du` inside the
# built image - it was the single largest thing in site-packages by 4x).
# ONNX Runtime does the same forward pass in about 50 MB, so the deployed
# image drops roughly 800 MB once torch and its sympy dependency are gone.
#
# The trade is that ultralytics' pre/post-processing has to be reimplemented
# here: letterbox resize in, box decode plus NMS out. Both are standard and
# short, and test_detect_parity.py checks the results against the PyTorch
# model on real camera frames so any drift shows up as a failing test rather
# than as quietly wrong congestion scores.
#
# The .pt weights stay in the repo. Training, re-export and any GPU work all
# still go through ultralytics; this module is only the inference path.

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ONNX_PATH = Path("weights/yolov12s.onnx")

# Ultralytics' predict defaults, mirrored so detections match the .pt model.
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.70
INPUT_SIZE = 640

# Grey used by ultralytics' letterbox padding. The value matters: the model
# saw this exact shade around letterboxed images during training.
PAD_COLOUR = (114, 114, 114)


class Detection:
    """One detected object, in original-image pixel coordinates."""

    __slots__ = ("cls_id", "name", "conf", "xyxy")

    def __init__(self, cls_id: int, name: str, conf: float, xyxy: tuple):
        self.cls_id = cls_id
        self.name = name
        self.conf = conf
        self.xyxy = xyxy  # (x1, y1, x2, y2)

    def __repr__(self) -> str:
        return f"Detection({self.name}, conf={self.conf:.2f})"


def _letterbox(frame: np.ndarray, size: int = INPUT_SIZE) -> tuple:
    """Resize to size x size while preserving aspect ratio, padding the rest.

    Returns (padded_image, scale, pad_x, pad_y) so detections can be mapped
    back to the original frame afterwards. Stretching instead of padding
    would distort the image and measurably shift the boxes.
    """
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Centre the image in the square, same as ultralytics.
    pad_x = (size - new_w) / 2
    pad_y = (size - new_h) / 2
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOUR
    )
    return padded, scale, left, top


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression; returns kept indices, best score first.

    Written out rather than pulled from a library because it is a dozen lines
    and avoids another dependency in an image we are explicitly shrinking.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break

        rest = order[1:]
        # Intersection of the best box with every remaining box.
        xx1 = np.maximum(x1[best], x1[rest])
        yy1 = np.maximum(y1[best], y1[rest])
        xx2 = np.minimum(x2[best], x2[rest])
        yy2 = np.minimum(y2[best], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)

        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]

    return keep


class Detector:
    """YOLOv12 detector backed by ONNX Runtime.

    Load once and reuse - creating the session parses the whole model, which
    takes a second or two.
    """

    def __init__(self, onnx_path: str | Path = ONNX_PATH):
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} missing - run `python -m routing.detect --export` "
                "once to convert weights/yolov12s.pt."
            )

        self.session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Ultralytics writes the class map into the ONNX metadata on export,
        # so the names travel with the model instead of being hardcoded here.
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta["names"])

    def detect(
        self,
        frame: np.ndarray,
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
    ) -> list[Detection]:
        """Detect objects in a BGR frame (the format OpenCV hands back)."""
        padded, scale, pad_x, pad_y = _letterbox(frame)

        # BGR uint8 HWC -> RGB float32 CHW, scaled to 0..1, with batch axis.
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        raw = self.session.run(None, {self.input_name: blob})[0]

        # Output is (1, 84, 8400): 4 box coords then 80 class scores, for
        # 8400 candidate positions. Transpose so each row is one candidate.
        preds = raw[0].T  # (8400, 84)

        class_scores = preds[:, 4:]
        best_cls = class_scores.argmax(axis=1)
        best_conf = class_scores.max(axis=1)

        keep_mask = best_conf >= conf
        if not keep_mask.any():
            return []

        preds = preds[keep_mask]
        best_cls = best_cls[keep_mask]
        best_conf = best_conf[keep_mask]

        # Boxes come back as centre-x, centre-y, width, height in the padded
        # 640x640 space. Convert to corners, then undo padding and scaling.
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = (cx - bw / 2 - pad_x) / scale
        y1 = (cy - bh / 2 - pad_y) / scale
        x2 = (cx + bw / 2 - pad_x) / scale
        y2 = (cy + bh / 2 - pad_y) / scale

        h, w = frame.shape[:2]
        x1 = np.clip(x1, 0, w)
        y1 = np.clip(y1, 0, h)
        x2 = np.clip(x2, 0, w)
        y2 = np.clip(y2, 0, h)
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # NMS per class, so an overlapping car and bus both survive.
        results: list[Detection] = []
        for cls_id in np.unique(best_cls):
            sel = best_cls == cls_id
            cls_boxes, cls_conf = boxes[sel], best_conf[sel]
            for i in _nms(cls_boxes, cls_conf, iou):
                results.append(
                    Detection(
                        cls_id=int(cls_id),
                        name=self.names.get(int(cls_id), str(cls_id)),
                        conf=float(cls_conf[i]),
                        xyxy=tuple(float(v) for v in cls_boxes[i]),
                    )
                )

        results.sort(key=lambda d: d.conf, reverse=True)
        return results

    def class_counts(self, frame: np.ndarray, **kwargs) -> dict[str, int]:
        """{class name: count} for one frame - what congestion scoring needs."""
        counts: dict[str, int] = {}
        for det in self.detect(frame, **kwargs):
            counts[det.name] = counts.get(det.name, 0) + 1
        return counts

    def annotate(self, frame: np.ndarray, dets: list[Detection] | None = None) -> np.ndarray:
        """Draw boxes and labels on a copy of the frame.

        Replaces ultralytics' Results.plot(), which is not available once
        torch is out of the image.
        """
        if dets is None:
            dets = self.detect(frame)

        out = frame.copy()
        for det in dets:
            x1, y1, x2, y2 = (int(v) for v in det.xyxy)
            # Stable per-class colour so the same class looks the same twice.
            colour = tuple(int(c) for c in _class_colour(det.cls_id))
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            label = f"{det.name} {det.conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Keep the label inside the frame when the box touches the top.
            top = max(y1, th + baseline + 2)
            cv2.rectangle(out, (x1, top - th - baseline - 2), (x1 + tw, top), colour, -1)
            cv2.putText(
                out, label, (x1, top - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return out


def _class_colour(cls_id: int) -> tuple:
    """Deterministic BGR colour per class id (no palette table needed)."""
    h = (cls_id * 47) % 180  # spread hues around the circle
    hsv = np.uint8([[[h, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


_DETECTOR: Detector | None = None


def get_detector(onnx_path: str | Path = ONNX_PATH) -> Detector:
    """Process-wide detector, created on first use."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Detector(onnx_path)
    return _DETECTOR


def export_from_pt(pt_path: str = "weights/yolov12s.pt") -> str:
    """Convert the PyTorch weights to ONNX. Needs torch + ultralytics, so it
    runs on a development machine, not inside the slim container."""
    from ultralytics import YOLO

    return YOLO(pt_path).export(format="onnx", imgsz=INPUT_SIZE, opset=13)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX YOLOv12 detector")
    parser.add_argument("--export", action="store_true",
                        help="re-export weights/yolov12s.pt to ONNX")
    args = parser.parse_args()

    if args.export:
        print("Exported:", export_from_pt())
        raise SystemExit

    # Self-test on a real camera frame, falling back to a synthetic image so
    # this still runs with no network (Docker build checks, CI).
    import pandas as pd
    import requests

    det = get_detector()
    print(f"Loaded {ONNX_PATH} with {len(det.names)} classes")

    frame = None
    try:
        cams = pd.read_csv("manhattan_cameras.csv")
        r = requests.get(cams.iloc[0]["image_url"], timeout=8)
        r.raise_for_status()
        frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        print(f"Live frame from: {cams.iloc[0]['name']}")
    except Exception as exc:
        print(f"No live frame ({type(exc).__name__}), using a synthetic image")

    if frame is None:
        frame = np.full((480, 640, 3), 120, np.uint8)

    dets = det.detect(frame)
    print(f"Detections: {len(dets)} -> {det.class_counts(frame)}")

    annotated = det.annotate(frame, dets)
    assert annotated.shape == frame.shape, "annotate must not change frame size"
    print("detect.py self-test OK")
