"""Arm/disarm test — bench test only, NO PROPELLERS.

Runs through arm sequence to verify Pixhawk + MAVLink wiring is correct.
Disarms after 5 seconds.

DO NOT RUN WITH PROPELLERS ON.

Usage:
    python scripts/arm_test.py [--sitl]
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.sitl:
        cfg['mavlink']['connection'] = 'udpin:127.0.0.1:14550'

    print("\n" + "=" * 60)
    print("WARNING: Make sure propellers are REMOVED before continuing!")
    print("=" * 60)
    response = input("Type 'YES' to confirm propellers are off: ")
    if response.strip() != 'YES':
        print("Aborted.")
        return

    mav = MAVLinkClient(
        connection_string=cfg['mavlink']['connection'],
        baud=cfg['mavlink']['baud'],
    )
    mav.connect(timeout=15)

    time.sleep(2)
    state = mav.get_state()
    print(f"\nInitial state: mode={state['mode']}, armed={state['armed']}")

    print("\nSwitching to GUIDED...")
    mav.set_mode('GUIDED')
    time.sleep(1)

    print("\nArming...")
    if not mav.arm(timeout=10):
        print("FAILED to arm")
    else:
        print("Armed! Holding for 5 seconds...")
        for i in range(5, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

    print("\nDisarming...")
    mav.disarm()
    time.sleep(2)

    state = mav.get_state()
    print(f"Final state: armed={state['armed']}")

    mav.close()


if __name__ == '__main__':
    main()
