#!/usr/bin/env python3
"""Test DDS communication with the Unitree G1 robot.

Subscribes to rt/lowstate for a few seconds and prints received data.
If nothing prints, DDS is not working (check cable, interface, power).

Usage:
    python scripts/tests/test_dds.py --interface en8
    python scripts/tests/test_dds.py --interface enp3s0 --duration 10
"""
from __future__ import annotations

import argparse
import sys
import time

from unitree_launcher.compat import patch_unitree_b2_import, patch_unitree_threading

patch_unitree_b2_import()
patch_unitree_threading()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


def main():
    parser = argparse.ArgumentParser(
        description="Test DDS communication with the G1 robot",
    )
    parser.add_argument(
        "--interface", required=True,
        help="Network interface (e.g. en8, enp3s0)",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0,
        help="How long to listen in seconds (default: 5)",
    )
    parser.add_argument(
        "--domain-id", type=int, default=0,
        help="DDS domain ID (default: 0)",
    )
    args = parser.parse_args()

    print(f"Initializing DDS on interface={args.interface}, domain={args.domain_id}")
    ChannelFactoryInitialize(args.domain_id, args.interface)

    msg_count = 0
    last_msg = None

    def on_message(msg):
        nonlocal msg_count, last_msg
        msg_count += 1
        last_msg = msg
        if msg_count <= 3 or msg_count % 100 == 0:
            q0 = msg.motor_state[0].q
            q3 = msg.motor_state[3].q
            mode = msg.mode_machine
            print(
                f"  [{msg_count:>5d}] mode_machine={mode}  "
                f"left_hip_pitch={q0:+.3f}  left_knee={q3:+.3f}"
            )

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_message, 1)

    print(f"Listening on rt/lowstate for {args.duration:.0f}s ...")
    print()

    start = time.time()
    while time.time() - start < args.duration:
        time.sleep(0.1)

    print()
    if msg_count == 0:
        print("NO MESSAGES RECEIVED.")
        print()
        print("Troubleshooting:")
        print(f"  - Is the cable plugged into interface {args.interface}?")
        print(f"  - Is the robot powered on and booted? (wait ~60s)")
        print(f"  - Run: ping 192.168.123.161")
        print(f"  - Check IP: ifconfig {args.interface} | grep inet")
        sys.exit(1)
    else:
        hz = msg_count / args.duration
        print(f"Received {msg_count} messages in {args.duration:.0f}s ({hz:.0f} Hz)")
        if last_msg is not None:
            print(f"  mode_machine = {last_msg.mode_machine}")
            print(f"  Motor states (first 6 joints):")
            for i in range(6):
                ms = last_msg.motor_state[i]
                print(f"    [{i:>2d}] q={ms.q:+.4f}  dq={ms.dq:+.4f}  tau={ms.tau_est:+.4f}")
        print()
        print("DDS is working.")


if __name__ == "__main__":
    main()
