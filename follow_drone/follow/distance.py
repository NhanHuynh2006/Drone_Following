"""Pinhole distance estimation from bbox height.

Z = (f_y * H_real) / h_pixels

Reference: NeoARCADE (arXiv 2504.01988, 2025) reports MAPE 1.94% at 1.1-7.5m
using exactly this formula with proper calibration.
"""

import numpy as np


class PinholeDistanceEstimator:
    def __init__(self, fy, person_height_m=1.7,
                 min_bbox_height_px=20, max_distance_m=30.0):
        self.fy = fy
        self.person_height_m = person_height_m
        self.min_bbox_height_px = min_bbox_height_px
        self.max_distance_m = max_distance_m

    def estimate(self, bbox):
        """Estimate distance from bbox [x1,y1,x2,y2].

        Returns distance in meters or None if unreliable.
        """
        x1, y1, x2, y2 = bbox
        h_px = y2 - y1
        if h_px < self.min_bbox_height_px:
            return None
        d = (self.fy * self.person_height_m) / h_px
        if d > self.max_distance_m or d < 0.5:
            return None
        return float(d)

    def variance(self, distance):
        """Approximate variance of distance estimate.

        Sai số tăng theo distance² vì 1 pixel error ở xa = nhiều mét.
        """
        if distance is None:
            return float('inf')
        # σ² ≈ (d / fy)² × (pixel_error)², với pixel_error ≈ 2
        sigma = (distance / self.fy) * 2 * distance
        return max(0.1, sigma ** 2)


class DistanceSmoother:
    """1D Kalman filter for distance smoothing."""

    def __init__(self, initial_distance=5.0, process_noise=0.5,
                 measurement_noise=1.0):
        self.x = initial_distance
        self.v = 0.0  # rate of change
        self.P = np.eye(2) * 5.0
        self.Q = np.array([[0.1, 0], [0, process_noise]])
        self.R = measurement_noise
        self.last_time = None

    def update(self, measurement, current_time, measurement_var=None):
        if self.last_time is None:
            self.last_time = current_time
            if measurement is not None:
                self.x = measurement
            return self.x

        dt = current_time - self.last_time
        self.last_time = current_time

        # Predict
        F = np.array([[1, dt], [0, 1]])
        state = np.array([self.x, self.v])
        state = F @ state
        self.P = F @ self.P @ F.T + self.Q

        if measurement is not None:
            # Update
            R = measurement_var if measurement_var is not None else self.R
            H = np.array([[1, 0]])
            y = measurement - H @ state
            S = H @ self.P @ H.T + R
            K = (self.P @ H.T) / S
            state = state + (K * y).flatten()
            self.P = (np.eye(2) - K @ H) @ self.P

        self.x, self.v = state[0], state[1]
        return self.x
