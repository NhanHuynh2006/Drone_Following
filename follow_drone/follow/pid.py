"""PID controllers for drone follow.

Three independent PIDs:
  - ForwardPID: distance error → forward velocity (body X)
  - YawPID:     image-space angular error → yaw rate
  - VerticalPID: altitude error → vertical velocity (NED Z, negative=up)
"""

import time


def _clip(x, low, high):
    return max(low, min(high, x))


class PID:
    """Generic PID controller with anti-windup."""

    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0,
                 output_min=-float('inf'), output_max=float('inf'),
                 integral_min=-float('inf'), integral_max=float('inf'),
                 setpoint=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_min = integral_min
        self.integral_max = integral_max
        self.setpoint = setpoint

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, measurement, current_time=None):
        if current_time is None:
            current_time = time.time()

        if self.prev_time is None:
            dt = 0.1
        else:
            dt = current_time - self.prev_time
            if dt <= 0:
                dt = 0.001
            if dt > 1.0:
                # Big gap (paused?) — reset to avoid integral windup
                self.reset()
                dt = 0.1
        self.prev_time = current_time

        error = self.setpoint - measurement

        # P
        P = self.Kp * error

        # I (with anti-windup clamp)
        self.integral += error * dt
        self.integral = _clip(self.integral, self.integral_min, self.integral_max)
        I = self.Ki * self.integral

        # D (filtered to reduce noise)
        D_raw = (error - self.prev_error) / dt
        D = self.Kd * D_raw
        self.prev_error = error

        output = P + I + D
        return _clip(output, self.output_min, self.output_max)


class ForwardPID(PID):
    """Distance to target → forward velocity (m/s in body X).

    Sign convention: distance > setpoint → tiến tới (positive vel)
                     distance < setpoint → lùi lại (negative vel)
    """

    def __init__(self, setpoint_distance_m=5.0, Kp=0.5, Ki=0.05, Kd=0.1,
                 v_max=3.0, integral_max=2.0):
        super().__init__(
            Kp=Kp, Ki=Ki, Kd=Kd,
            output_min=-v_max, output_max=v_max,
            integral_min=-integral_max, integral_max=integral_max,
            setpoint=setpoint_distance_m,
        )

    def compute(self, distance_m, current_time=None):
        # Note: error = setpoint - measurement. We want vel = -(error)
        # because distance > setpoint means we need to MOVE FORWARD (positive vel).
        # So we negate the standard PID output.
        return -super().compute(distance_m, current_time)


class YawPID:
    """Bbox horizontal center → yaw rate (rad/s).

    Sign convention: bbox right of center → positive yaw rate (turn right).
    """

    def __init__(self, image_width, fx, Kp=2.0, Kd=0.3, rate_max=1.5):
        self.image_width = image_width
        self.fx = fx
        self.Kp = Kp
        self.Kd = Kd
        self.rate_max = rate_max
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, bbox_cx, current_time=None):
        if current_time is None:
            current_time = time.time()

        if self.prev_time is None:
            dt = 0.1
        else:
            dt = current_time - self.prev_time
            if dt <= 0:
                dt = 0.001
            if dt > 1.0:
                self.prev_error = 0.0
                dt = 0.1
        self.prev_time = current_time

        error_px = bbox_cx - self.image_width / 2
        error_rad = error_px / self.fx

        P = self.Kp * error_rad
        D = self.Kd * (error_rad - self.prev_error) / dt
        self.prev_error = error_rad

        # Positive error_rad (target right of center) → positive yaw rate (turn right)
        yaw_rate = P + D
        return _clip(yaw_rate, -self.rate_max, self.rate_max)

    def reset(self):
        self.prev_error = 0.0
        self.prev_time = None


class VerticalPID(PID):
    """Altitude AGL → vertical velocity (m/s, positive = up in body frame).

    Note: NED has Z down, so we negate when sending to MAVLink.
    """

    def __init__(self, setpoint_altitude_m=5.0, Kp=0.8, Ki=0.1, Kd=0.2,
                 v_max=2.0, integral_max=1.0):
        super().__init__(
            Kp=Kp, Ki=Ki, Kd=Kd,
            output_min=-v_max, output_max=v_max,
            integral_min=-integral_max, integral_max=integral_max,
            setpoint=setpoint_altitude_m,
        )
