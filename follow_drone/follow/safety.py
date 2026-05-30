"""Safety / Failsafe handler.

Implements the 3-tier target-lost ladder:
  0-3s lost:   coast (continue with Kalman prediction)
  3-8s lost:   hover (velocity = 0)
  8-15s lost:  loiter (switch FCU mode to LOITER)
  >15s lost:   RTL (Return To Launch)

Plus various pre-flight and runtime checks.
"""

import time


class SafetyState:
    FOLLOW = 'follow'
    COAST = 'coast'
    HOVER = 'hover'
    LOITER = 'loiter'
    RTL = 'rtl'
    EMERGENCY = 'emergency'


class SafetyManager:
    def __init__(self, mavlink_client, config):
        self.mav = mavlink_client
        self.cfg = config

        self.target_lost_since = None
        self.current_state = SafetyState.FOLLOW
        self.last_command_time = time.time()

        self.coast_duration = config['target_lost']['coast_duration_s']
        self.hover_duration = config['target_lost']['hover_duration_s']
        self.loiter_duration = config['target_lost']['loiter_duration_s']
        # cumulative thresholds
        self.t_coast_end = self.coast_duration
        self.t_hover_end = self.coast_duration + self.hover_duration
        self.t_loiter_end = self.t_hover_end + self.loiter_duration

    # ───── Pre-flight checks ─────────────────────────────────────

    def preflight_check(self):
        """Run all pre-flight checks. Returns (ok, list_of_warnings)."""
        warnings = []
        ok = True
        state = self.mav.get_state()

        # GPS
        if self.cfg.get('require_gps_3d_fix', True):
            if state['gps_fix'] < 3:
                warnings.append(f"GPS fix < 3D (current: {state['gps_fix']})")
                ok = False
            if state['gps_eph'] > self.cfg.get('max_hdop', 200):
                warnings.append(f"GPS HDOP too high: {state['gps_eph']/100:.2f}")
                ok = False
            if state['gps_sats'] < 6:
                warnings.append(f"GPS sats < 6 (current: {state['gps_sats']})")

        # Battery
        if state['battery_voltage'] is not None and state['battery_voltage'] < 22.5:
            warnings.append(f"Battery low: {state['battery_voltage']:.1f}V")
            ok = False

        # Heartbeat
        hb_age = self.mav.heartbeat_age()
        if hb_age > 2.0:
            warnings.append(f"Heartbeat age too high: {hb_age:.1f}s")
            ok = False

        # Home position
        if state['home_position'] is None:
            warnings.append("No HOME position set")

        return ok, warnings

    # ───── Target-lost ladder ────────────────────────────────────

    def update_target_status(self, target_present):
        """Call every loop iteration.

        Returns one of: 'follow', 'coast', 'hover', 'loiter', 'rtl'
        """
        now = time.time()

        if target_present:
            if self.target_lost_since is not None:
                print(f"[Safety] Target re-acquired after {now - self.target_lost_since:.1f}s")
            self.target_lost_since = None
            self.current_state = SafetyState.FOLLOW
            return SafetyState.FOLLOW

        if self.target_lost_since is None:
            self.target_lost_since = now
            return SafetyState.COAST

        elapsed = now - self.target_lost_since
        if elapsed < self.t_coast_end:
            new_state = SafetyState.COAST
        elif elapsed < self.t_hover_end:
            new_state = SafetyState.HOVER
        elif elapsed < self.t_loiter_end:
            new_state = SafetyState.LOITER
        else:
            new_state = SafetyState.RTL

        if new_state != self.current_state:
            print(f"[Safety] State change: {self.current_state} → {new_state} "
                  f"(target lost {elapsed:.1f}s)")
            if new_state == SafetyState.LOITER:
                self.mav.set_mode('LOITER')
            elif new_state == SafetyState.RTL:
                self.mav.set_mode('RTL')
            self.current_state = new_state

        return new_state

    # ───── Velocity safety clamps ────────────────────────────────

    def clamp_velocity(self, vx, vy, vz, yaw_rate):
        v_max = self.cfg['v_max_horizontal']
        yr_max = self.cfg['yaw_rate_max']
        return (
            _clip(vx, -v_max, v_max),
            _clip(vy, -v_max, v_max),
            _clip(vz, -2.0, 2.0),
            _clip(yaw_rate, -yr_max, yr_max),
        )

    def clamp_forward_by_distance(self, vx_body, distance):
        """Don't get too close to person."""
        if distance is None:
            return vx_body
        d_min = self.cfg['d_min']
        d_max = self.cfg['d_max']
        if distance < d_min:
            # Too close — force backward at fixed pace
            return min(vx_body, -0.5)
        if distance > d_max:
            # Too far — limit forward speed (detection unreliable)
            return _clip(vx_body, 0.0, 1.0)
        return vx_body

    def clamp_altitude(self, vz_body, current_altitude):
        """Don't go below min or above max altitude."""
        alt_min = self.cfg['altitude_min_agl']
        alt_max = self.cfg['altitude_max_agl']
        if current_altitude < alt_min and vz_body > 0:
            # vz_body positive = down (NED) — if too low, refuse to descend
            return min(vz_body, 0.0)
        if current_altitude > alt_max and vz_body < 0:
            # If too high, refuse to climb
            return max(vz_body, 0.0)
        return vz_body

    # ───── Continuous health check ───────────────────────────────

    def runtime_health_check(self):
        """Check during follow. Returns (ok, reason_if_not)."""
        state = self.mav.get_state()

        # Heartbeat
        hb_age = self.mav.heartbeat_age()
        if hb_age > 3.0:
            return False, f"Heartbeat lost ({hb_age:.1f}s)"

        # GPS dropout
        if state['gps_fix'] < 3:
            return False, f"GPS fix lost (fix={state['gps_fix']})"

        # Battery
        if state['battery_voltage'] is not None:
            if state['battery_voltage'] < 21.5:
                return False, f"Battery critical: {state['battery_voltage']:.1f}V"

        # Altitude sanity
        alt = state.get('altitude_agl', 0)
        if alt > self.cfg['altitude_max_agl'] * 1.5:
            return False, f"Altitude exceeded: {alt:.1f}m"

        return True, None


def _clip(x, low, high):
    return max(low, min(high, x))
