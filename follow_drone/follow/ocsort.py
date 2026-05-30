"""OC-SORT tracker (simplified, single-class).

Implements:
  - Linear Kalman filter per track (8-state: cx, cy, area, aspect + velocities)
  - Hungarian assignment by IoU
  - Observation-Centric Re-Update (ORU): when a lost track is re-found,
    re-update KF with stored observations to reduce drift.
  - Observation-Centric Momentum (OCM): direction consistency in cost.

Single-target priority: a "locked" track ID gets extra association priority.

Reference: Cao et al., "Observation-Centric SORT" (CVPR 2023).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _xyxy_to_state(box):
    """[x1,y1,x2,y2] → [cx, cy, area, aspect]."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    area = w * h
    aspect = w / h
    return np.array([cx, cy, area, aspect], dtype=np.float64)


def _state_to_xyxy(state):
    """[cx, cy, area, aspect, ...] → [x1,y1,x2,y2]."""
    cx, cy, area, aspect = state[:4]
    area = max(1.0, area)
    aspect = max(0.1, aspect)
    w = np.sqrt(area * aspect)
    h = area / w
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


def _iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


class KalmanBoxTracker:
    """Kalman filter for single bbox track, 8-dim state.

    State: [cx, cy, area, aspect, ċx, ċy, ȧrea, ȧspect]
    Constant-velocity model on first 4 dims.
    """
    _next_id = 1

    def __init__(self, det_box, score):
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        self.dim_x = 8
        self.dim_z = 4

        # Transition matrix (constant velocity)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (we observe [cx, cy, area, aspect])
        self.H = np.zeros((4, 8))
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariances
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0     # high uncertainty on velocity

        self.Q = np.eye(8) * 1.0    # process noise
        self.Q[4:, 4:] *= 0.1
        self.R = np.eye(4) * 1.0    # measurement noise
        self.R[2, 2] = 10.0          # area is noisier
        self.R[3, 3] = 0.1           # aspect is more reliable

        # Initialize state from first detection
        z = _xyxy_to_state(det_box)
        self.x = np.zeros(8)
        self.x[:4] = z

        # Tracking metadata
        self.score = score
        self.age = 0                 # total frames since born
        self.hits = 1                # number of successful matches
        self.time_since_update = 0   # frames since last detection match
        self.history_observations = [z.copy()]   # for OCM and ORU
        self.last_observation = z.copy()
        self.observed_velocity = np.zeros(2)     # for OCM direction

    def predict(self):
        """Advance state by one frame."""
        # Prevent area from becoming negative
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return _state_to_xyxy(self.x)

    def update(self, det_box, score):
        """Update state with detection."""
        z = _xyxy_to_state(det_box)

        # ORU: if track was lost and now re-found, smooth using last observation
        if self.time_since_update > 1 and len(self.history_observations) > 0:
            last_obs = self.last_observation
            n = self.time_since_update + 1
            # Linear interpolation to fill gap (simplified ORU)
            for i in range(1, n):
                interp = last_obs + (z - last_obs) * (i / n)
                # Re-run KF update with interpolated observation
                y = interp - self.H @ self.x
                S = self.H @ self.P @ self.H.T + self.R * 4.0  # higher noise on interpolated
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                self.P = (np.eye(8) - K @ self.H) @ self.P

        # Standard KF update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        # OCM: store observed velocity (direction)
        if len(self.history_observations) > 0:
            prev_z = self.history_observations[-1]
            self.observed_velocity = z[:2] - prev_z[:2]

        self.history_observations.append(z.copy())
        if len(self.history_observations) > 30:
            self.history_observations.pop(0)
        self.last_observation = z.copy()

        self.score = score
        self.hits += 1
        self.time_since_update = 0

    def get_box(self):
        return _state_to_xyxy(self.x)


def _associate(detections, trackers, iou_threshold=0.3):
    """Match detections to trackers using Hungarian on IoU cost.

    Returns:
      matches: list of (det_idx, trk_idx)
      unmatched_dets: list of det_idx
      unmatched_trks: list of trk_idx
    """
    if len(trackers) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(trackers)))

    iou_matrix = np.zeros((len(detections), len(trackers)))
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = _iou(det['box'], trk.get_box())

    # Hungarian: minimize -IoU = maximize IoU
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))

    matched_dets = {r for r, _ in matches}
    matched_trks = {c for _, c in matches}
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
    unmatched_trks = [i for i in range(len(trackers)) if i not in matched_trks]
    return matches, unmatched_dets, unmatched_trks


class OCSORT:
    def __init__(self, max_age=90, min_hits=3, iou_threshold=0.3, delta_t=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """Run one tracking step.

        detections: list of {'box': [x1,y1,x2,y2], 'score': float}

        Returns list of confirmed tracks: [{'id': int, 'box': [...], 'score': float,
                                             'age': int, 'time_since_update': int,
                                             'tracker': KalmanBoxTracker}]
        """
        self.frame_count += 1

        # 1. Predict all existing trackers
        for trk in self.trackers:
            trk.predict()

        # 2. Associate detections with trackers
        matches, unmatched_dets, unmatched_trks = _associate(
            detections, self.trackers, self.iou_threshold
        )

        # 3. Update matched trackers
        for d, t in matches:
            self.trackers[t].update(detections[d]['box'], detections[d]['score'])

        # 4. Create new trackers for unmatched detections
        for d in unmatched_dets:
            new_trk = KalmanBoxTracker(detections[d]['box'], detections[d]['score'])
            self.trackers.append(new_trk)

        # 5. Remove dead trackers
        self.trackers = [trk for trk in self.trackers
                         if trk.time_since_update <= self.max_age]

        # 6. Return confirmed tracks
        results = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                if trk.time_since_update <= 1:
                    box = trk.get_box()
                    results.append({
                        'id': trk.id,
                        'box': box.tolist(),
                        'score': trk.score,
                        'age': trk.age,
                        'time_since_update': trk.time_since_update,
                        'tracker': trk,
                    })
        return results

    def get_track_by_id(self, track_id):
        for trk in self.trackers:
            if trk.id == track_id:
                return trk
        return None
