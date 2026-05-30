"""Test MAVLink connection without flying.

Connects, prints heartbeat info, drone state, then disconnects.
NO MOTOR ARMING. Safe to run anytime.

Usage:
    python scripts/test_mavlink.py [--sitl]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from follow.mavlink_client import MAVLinkClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--sitl', action='store_true')
    parser.add_argument('--duration', type=float, default=30.0,
                        help="How long to monitor (seconds)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.sitl:
        cfg['mavlink']['connection'] = 'udpin:127.0.0.1:14550'

    print(f"Connecting to {cfg['mavlink']['connection']}...")
    mav = MAVLinkClient(
        connection_string=cfg['mavlink']['connection'],
        baud=cfg['mavlink']['baud'],
    )
    mav.connect(timeout=15)

    print(f"\nMonitoring for {args.duration}s. Ctrl+C to stop.\n")
    t0 = time.time()
    last_print = 0

    try:
        while time.time() - t0 < args.duration:
            now = time.time()
            if now - last_print > 1.0:
                state = mav.get_state()
                pos = state.get('local_position')
                att = state.get('attitude')
                print(f"[{now-t0:5.1f}s] mode={state.get('mode')} "
                      f"armed={state.get('armed')} "
                      f"pos={pos} "
                      f"alt_agl={state.get('altitude_agl', 0):.1f}m "
                      f"yaw={att[2] if att else 0:.2f}rad "
                      f"bat={state.get('battery_voltage')}V "
                      f"gps_fix={state.get('gps_fix')} "
                      f"sats={state.get('gps_sats')} "
                      f"hb_age={mav.heartbeat_age():.1f}s")
                last_print = now
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        mav.close()


if __name__ == '__main__':
    main()
