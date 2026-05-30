"""Hybrid 2.5D Visual Servoing.

Combines:
  - IBVS (image-based) for yaw — use bbox horizontal center
  - PBVS (position-based) for forward range — use estimated distance
  - Altitude control via PID on AGL

Output: (vx_body, vy_body, vz_body, yaw_rate) ready to send to MAVLink.
"""

from .pid import ForwardPID, YawPID, VerticalPID


class VisualServo:
    def __init__(self, cfg, image_width, image_height, fx, fy):
        self.image_width = image_width
        self.image_height = image_height
        self.fx = fx
        self.fy = fy

        self.follow_distance_m = cfg['follow_distance_m']
        self.follow_altitude_m = cfg['follow_altitude_m']

        self.forward_pid = ForwardPID(
            setpoint_distance_m=self.follow_distance_m,
            **cfg['forward_pid'],
        )
        self.yaw_pid = YawPID(
            image_width=image_width,
            fx=fx,
            **cfg['yaw_pid'],
        )
        self.vertical_pid = VerticalPID(
            setpoint_altitude_m=self.follow_altitude_m,
            **cfg['vertical_pid'],
        )

    def reset(self):
        self.forward_pid.reset()
        self.yaw_pid.reset()
        self.vertical_pid.reset()

    def compute(self, bbox, distance_m, altitude_agl_m, current_time):
        """Compute body-frame velocity setpoint.

        bbox:           [x1, y1, x2, y2] of locked target
        distance_m:     estimated distance to target (from pinhole)
        altitude_agl_m: drone altitude AGL
        current_time:   timestamp

        Returns: (vx_body, vy_body, vz_body_ned, yaw_rate)
                 vz_body_ned: NED Z velocity (positive = down)
        """
        x1, y1, x2, y2 = bbox
        bbox_cx = (x1 + x2) / 2

        # Forward velocity: distance error → vx_body (forward)
        if distance_m is not None:
            vx_body = self.forward_pid.compute(distance_m, current_time)
        else:
            vx_body = 0.0  # No reliable distance, don't move forward

        # Lateral velocity: 0 (we use yaw to track laterally)
        vy_body = 0.0

        # Vertical velocity: altitude error → up/down
        # vertical_pid output > 0 means need to go UP
        vz_up = self.vertical_pid.compute(altitude_agl_m, current_time)
        # NED Z is down → flip sign
        vz_body_ned = -vz_up

        # Yaw rate: bbox horizontal error → yaw rate
        yaw_rate = self.yaw_pid.compute(bbox_cx, current_time)

        return vx_body, vy_body, vz_body_ned, yaw_rate
