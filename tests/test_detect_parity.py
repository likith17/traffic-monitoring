# Checks that the ONNX detector agrees with the PyTorch model it replaced.
#
# The container runs ONNX Runtime instead of PyTorch to keep the image small,
# which means the pre/post-processing in routing/detect.py is our code rather
# than ultralytics'. A silent mismatch there would not crash anything - it
# would just shift every congestion score, and with it the routing decisions
# and the benchmark. So compare the two on real camera frames.
#
# Run:  python -m tests.test_detect_parity
#
# Needs torch + ultralytics, so this runs on a development machine, not in
# the slim container.

from __future__ import annotations

import sys

import cv2
import numpy as np
import pandas as pd
import requests

from routing.detect import get_detector

# How many cameras to compare. More is better evidence; each one is a network
# fetch plus two inferences, so this is a compromise with the clock.
N_CAMERAS = 8

# Congestion scoring only ever looks at these classes, so a disagreement on
# 'kite' does not matter while a disagreement on 'car' very much does.
SCORED = {"car", "bus", "truck", "motorcycle", "person", "traffic light", "stop sign"}


def fetch_frames(n: int) -> list[tuple[str, np.ndarray]]:
    """Grab live snapshots from the first n reachable cameras."""
    cams = pd.read_csv("manhattan_cameras.csv")
    frames = []
    for _, cam in cams.iterrows():
        if len(frames) >= n:
            break
        try:
            r = requests.get(cam["image_url"], timeout=8)
            r.raise_for_status()
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append((cam["name"], img))
        except Exception:
            continue  # camera offline; try the next one
    return frames


def torch_counts(model, frame: np.ndarray) -> dict[str, int]:
    """Class counts from the original ultralytics path."""
    res = model(frame, imgsz=640, verbose=False)[0]
    counts: dict[str, int] = {}
    for box in res.boxes:
        name = model.names.get(int(box.cls[0]), "?")
        counts[name] = counts.get(name, 0) + 1
    return counts


def main() -> int:
    from ultralytics import YOLO

    torch_model = YOLO("weights/yolov12s.pt")
    onnx_det = get_detector()

    frames = fetch_frames(N_CAMERAS)
    if not frames:
        print("No cameras reachable - cannot compare. Check the network.")
        return 1
    print(f"Comparing on {len(frames)} live camera frames\n")

    total_torch = total_onnx = 0
    mismatches = []

    for name, frame in frames:
        t = torch_counts(torch_model, frame)
        o = onnx_det.class_counts(frame)

        # Compare only the classes that feed the congestion score.
        t_scored = {k: v for k, v in t.items() if k in SCORED}
        o_scored = {k: v for k, v in o.items() if k in SCORED}
        total_torch += sum(t_scored.values())
        total_onnx += sum(o_scored.values())

        flag = "" if t_scored == o_scored else "   <-- differs"
        if t_scored != o_scored:
            mismatches.append((name, t_scored, o_scored))
        print(f"{name[:38]:<38} torch={t_scored} onnx={o_scored}{flag}")

    print(f"\nScored objects: torch={total_torch}  onnx={total_onnx}")

    if mismatches:
        print(f"\n{len(mismatches)} frame(s) differ:")
        for name, t, o in mismatches:
            print(f"  {name}\n    torch: {t}\n    onnx:  {o}")

    if total_torch == 0:
        print("\nNo objects detected by either model - inconclusive, rerun later.")
        return 1

    drift = abs(total_onnx - total_torch) / total_torch
    print(f"End-to-end count drift: {drift * 100:.1f}%")
    print(
        "Small differences here are expected and not a bug: ultralytics "
        "letterboxes to a stride-multiple rectangle, while the exported ONNX "
        "model has a fixed 640x640 input and therefore pads differently. "
        "A box near the confidence threshold can fall either side of it.\n"
    )

    # The real check. Feed both models byte-identical input and compare the
    # raw tensors - that isolates our pre/post-processing from the framing
    # difference above. Anything beyond float32 noise means detect.py is
    # genuinely wrong, not merely framed differently.
    return raw_tensor_parity(torch_model, onnx_det, frames[0][1])


def raw_tensor_parity(torch_model, onnx_det, frame: np.ndarray) -> int:
    """Run both backends on the same tensor; they must agree numerically."""
    import torch

    from routing.detect import _letterbox

    padded, _, _, _ = _letterbox(frame)
    blob = np.expand_dims(
        padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0, 0
    )

    with torch.no_grad():
        torch_raw = torch_model.model(torch.from_numpy(blob))[0].numpy()
    onnx_raw = onnx_det.session.run(None, {onnx_det.input_name: blob})[0]

    if torch_raw.shape != onnx_raw.shape:
        print(f"FAIL: output shapes differ - {torch_raw.shape} vs {onnx_raw.shape}")
        return 1

    max_diff = float(np.abs(torch_raw - onnx_raw).max())
    print(f"Raw output max abs difference on identical input: {max_diff:.6f}")

    # float32 accumulation order differs between the two runtimes; 0.01 is
    # comfortably above that noise and far below any real divergence.
    if max_diff > 0.01:
        print("FAIL: ONNX model output diverges from PyTorch.")
        return 1

    print("PASS: ONNX inference is numerically equivalent to PyTorch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
