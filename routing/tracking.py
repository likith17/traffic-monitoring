# Multi-frame vehicle tracking, Phase 1 of the upgrade plan.
#
# A single camera frame gives an object count, and that count is noisy: one
# frame can catch a red light with a full queue and the next can catch an empty
# crossing. This module follows vehicles across several frames so we can measure
# things a single frame cannot: how many distinct vehicles pass (flow), and how
# many sit still (queue length). Those are steadier and more meaningful signals
# than a per-frame count.
#
# The tracker is a compact, ByteTrack-style associator. It keeps the paper's
# central idea, a two-stage match that uses low-confidence detections to hold on
# to vehicles during brief occlusion, but it is a readable re-implementation
# rather than a bit-exact port. It runs on the boxes our ONNX detector already
# produces (routing/detect.py), so it needs no deep-learning framework and adds
# no heavy dependency: association is a Hungarian match over IoU, solved with
# scipy, which the project already installs through osmnx.
#
# Honest scope. All motion here is measured in image pixels, not metres. Without
# per-camera calibration (a homography from image to ground plane, which the DOT
# feeds do not provide) pixel motion cannot be converted to km/h. So this module
# reports a *relative* speed proxy and a stationary/queued flag, and it is
# careful never to claim a real-world speed.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from routing.detect import Detection

# Classes that count as vehicles for flow and queue measurement. Pedestrians and
# signals are deliberately excluded here; they are handled by the scoring layer.
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}

# A track is confirmed only after it has been seen this many frames in a row.
# This suppresses one-frame false detections from becoming phantom vehicles.
MIN_HITS = 3

# A track is dropped after this many frames with no matching detection. Large
# enough to survive a brief occlusion, small enough not to keep ghosts around.
MAX_AGE = 30

# Minimum IoU for a detection to be allowed to match a track. Below this the
# boxes overlap too little to be the same vehicle.
MIN_IOU = 0.2

# Detections at or above this confidence drive the first association stage;
# those below it are used only to keep existing tracks alive (the ByteTrack idea).
HIGH_CONF = 0.5


def _iou_matrix(tracks_xyxy: np.ndarray, dets_xyxy: np.ndarray) -> np.ndarray:
    """IoU of every track box against every detection box, shape (T, D)."""
    if len(tracks_xyxy) == 0 or len(dets_xyxy) == 0:
        return np.zeros((len(tracks_xyxy), len(dets_xyxy)), dtype=np.float32)

    # Broadcast track boxes (T,1,4) against detection boxes (1,D,4).
    t = tracks_xyxy[:, None, :]
    d = dets_xyxy[None, :, :]

    inter_x1 = np.maximum(t[..., 0], d[..., 0])
    inter_y1 = np.maximum(t[..., 1], d[..., 1])
    inter_x2 = np.minimum(t[..., 2], d[..., 2])
    inter_y2 = np.minimum(t[..., 3], d[..., 3])

    inter = np.clip(inter_x2 - inter_x1, 0, None) * np.clip(inter_y2 - inter_y1, 0, None)
    area_t = (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    area_d = (d[..., 2] - d[..., 0]) * (d[..., 3] - d[..., 1])
    return inter / (area_t + area_d - inter + 1e-9)


@dataclass
class Track:
    """One vehicle followed across frames."""

    track_id: int
    cls_name: str
    xyxy: tuple                     # most recent box
    centroids: list = field(default_factory=list)  # (cx, cy) per matched frame
    hits: int = 1                   # frames matched, total
    time_since_update: int = 0      # frames since last match
    confirmed: bool = False

    @property
    def centroid(self) -> tuple:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def displacement(self) -> float:
        """Total path length in pixels over the track's life.

        Sum of frame-to-frame centroid moves, so a vehicle that crawls forward
        and one that idles are told apart even if they end near where they
        started.
        """
        if len(self.centroids) < 2:
            return 0.0
        pts = np.asarray(self.centroids)
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        return float(steps.sum())

    def mean_step_px(self) -> float:
        """Average per-frame pixel movement: the relative speed proxy."""
        if len(self.centroids) < 2:
            return 0.0
        return self.displacement() / (len(self.centroids) - 1)


class VehicleTracker:
    """ByteTrack-style multi-object tracker over detector output."""

    def __init__(
        self,
        min_iou: float = MIN_IOU,
        max_age: int = MAX_AGE,
        min_hits: int = MIN_HITS,
        high_conf: float = HIGH_CONF,
    ):
        self.min_iou = min_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self.high_conf = high_conf
        self.tracks: list[Track] = []
        self._next_id = 0

    def _match(self, tracks: list[Track], dets: list[Detection]) -> tuple:
        """Hungarian IoU match between a set of tracks and detections.

        Returns (pairs, unmatched_track_idx, unmatched_det_idx), where pairs are
        (track_index, det_index) with IoU at least min_iou.
        """
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        t_boxes = np.array([t.xyxy for t in tracks], dtype=np.float32)
        d_boxes = np.array([d.xyxy for d in dets], dtype=np.float32)
        iou = _iou_matrix(t_boxes, d_boxes)

        # Hungarian solver minimises cost, so use 1 - IoU.
        rows, cols = linear_sum_assignment(1.0 - iou)

        pairs, matched_t, matched_d = [], set(), set()
        for r, c in zip(rows, cols):
            if iou[r, c] >= self.min_iou:
                pairs.append((r, c))
                matched_t.add(r)
                matched_d.add(c)

        unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
        unmatched_d = [j for j in range(len(dets)) if j not in matched_d]
        return pairs, unmatched_t, unmatched_d

    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance the tracker by one frame and return the confirmed tracks."""
        vehicles = [d for d in detections if d.name in VEHICLE_CLASSES]
        high = [d for d in vehicles if d.conf >= self.high_conf]
        low = [d for d in vehicles if d.conf < self.high_conf]

        for t in self.tracks:
            t.time_since_update += 1

        # Stage 1: match high-confidence detections to all tracks.
        pairs, unmatched_t, unmatched_d_high = self._match(self.tracks, high)
        self._apply_matches(self.tracks, high, pairs)

        # Stage 2: the ByteTrack step. Try to keep still-unmatched tracks alive
        # using the low-confidence detections, which often are real vehicles
        # dimmed by occlusion or motion blur.
        remaining_tracks = [self.tracks[i] for i in unmatched_t]
        pairs2, unmatched_t2, _ = self._match(remaining_tracks, low)
        self._apply_matches(remaining_tracks, low, pairs2)

        # Any high-confidence detection still unmatched starts a new track. Low
        # detections never spawn tracks; they only sustain existing ones.
        for j in unmatched_d_high:
            det = high[j]
            self.tracks.append(
                Track(track_id=self._next_id, cls_name=det.name, xyxy=det.xyxy,
                      centroids=[_centroid(det.xyxy)])
            )
            self._next_id += 1

        # Confirm tracks that have been seen enough, drop stale ones.
        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.confirmed = True
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def _apply_matches(self, tracks: list[Track], dets: list[Detection], pairs) -> None:
        for ti, di in pairs:
            t, d = tracks[ti], dets[di]
            t.xyxy = d.xyxy
            t.centroids.append(_centroid(d.xyxy))
            t.hits += 1
            t.time_since_update = 0


def _centroid(xyxy: tuple) -> tuple:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class FlowStats:
    """Traffic-flow features derived from a tracked frame sequence."""

    n_frames: int
    unique_vehicles: int      # distinct confirmed tracks over the window (flow)
    queue_length: int         # confirmed tracks that stayed near-stationary
    mean_density: float       # average simultaneous vehicles per frame
    mean_speed_px: float      # relative speed proxy, pixels/frame (NOT km/h)

    def as_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "unique_vehicles": self.unique_vehicles,
            "queue_length": self.queue_length,
            "mean_density": round(self.mean_density, 2),
            "mean_speed_px": round(self.mean_speed_px, 2),
        }


def flow_stats_from(tracker: VehicleTracker, per_frame_counts, n_frames, stationary_px=2.0) -> FlowStats:
    """Turn a finished tracker plus per-frame vehicle counts into FlowStats.

    Separated out so callers that already run detection for another purpose (the
    segment analyser, which also averages per-frame scores) can reuse the same
    tracker without detecting every frame a second time.
    """
    confirmed = [t for t in tracker.tracks if t.confirmed]
    queued = sum(1 for t in confirmed if t.mean_step_px() < stationary_px)
    speeds = [t.mean_step_px() for t in confirmed if len(t.centroids) >= 2]
    return FlowStats(
        n_frames=n_frames,
        unique_vehicles=len(confirmed),
        queue_length=queued,
        mean_density=float(np.mean(per_frame_counts)) if per_frame_counts else 0.0,
        mean_speed_px=float(np.mean(speeds)) if speeds else 0.0,
    )


def track_sequence(
    frames,
    detector,
    stationary_px: float = 2.0,
    conf: float = 0.25,
) -> FlowStats:
    """Run detection and tracking over an iterable of BGR frames.

    stationary_px is the average per-frame pixel movement below which a
    confirmed track counts as queued. It is a threshold on the relative speed
    proxy, so it scales with frame rate and resolution and should be tuned per
    feed; the default suits the low-frame-rate DOT stills.
    """
    tracker = VehicleTracker()
    per_frame_counts: list[int] = []
    n = 0

    for frame in frames:
        n += 1
        dets = detector.detect(frame, conf=conf)
        tracker.update(dets)
        per_frame_counts.append(
            sum(1 for d in dets if d.name in VEHICLE_CLASSES and d.conf >= HIGH_CONF)
        )

    return flow_stats_from(tracker, per_frame_counts, n, stationary_px)


def track_video(video_path, detector, stride: int = 1, max_frames: int = 300) -> FlowStats:
    """Convenience wrapper: track a video file on disk.

    stride>1 subsamples frames; note that skipping frames makes each vehicle
    appear to move further per step, so a larger stride needs a larger
    stationary_px threshold to keep the queue definition meaningful.
    """
    import cv2

    def frame_iter():
        cap = cv2.VideoCapture(str(video_path))
        idx = yielded = 0
        try:
            while yielded < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    yielded += 1
                    yield frame
                idx += 1
        finally:
            cap.release()

    return track_sequence(frame_iter(), detector)


if __name__ == "__main__":
    # Self-test on a synthetic sequence, so the tracker's mechanics can be
    # checked deterministically with no video file and no network. Three
    # "vehicles" move left to right at different speeds; a fourth sits still to
    # exercise the queue detector. We feed detector-style boxes straight in,
    # bypassing the ONNX model, since here we are testing association, not
    # detection.
    from routing.detect import Detection

    W, H, N = 640, 480, 40

    class _ScriptedDetector:
        """Returns pre-scripted boxes per frame, mimicking Detector.detect."""

        def __init__(self):
            self.frame = 0

        def detect(self, _frame, conf=0.25):
            f = self.frame
            self.frame += 1
            dets = []
            # Three movers at 6, 4, and 8 px/frame.
            for lane, speed in enumerate((6, 4, 8)):
                x = 40 + speed * f
                if x < W - 40:
                    y = 100 + lane * 100
                    dets.append(Detection(2, "car", 0.85, (x, y, x + 40, y + 30)))
            # One stationary vehicle (a queue of one).
            dets.append(Detection(2, "car", 0.80, (300, 400, 340, 430)))
            return dets

    stats = track_sequence((None for _ in range(N)), _ScriptedDetector(), stationary_px=2.0)
    print("Flow stats:", stats.as_dict())

    assert stats.unique_vehicles == 4, f"expected 4 vehicles, got {stats.unique_vehicles}"
    assert stats.queue_length == 1, f"expected 1 queued, got {stats.queue_length}"
    assert stats.mean_speed_px > 2.0, "movers should register motion above the queue threshold"
    print("tracking.py self-test OK")
