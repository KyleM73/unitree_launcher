#!/usr/bin/env python3
"""Interactive gamepad mapping tool for Logitech F310 (D-Input mode).

Walks through each button and axis, asks you to press/move it,
and prints the raw HID bytes so we can verify the mapping.

Usage:
    uv run python scripts/tests/gamepad_mapping.py
"""
import hid
import time
import sys

VID = 0x046D
PID = 0xC216


def open_gamepad():
    devices = hid.enumerate(VID, PID)
    if not devices:
        print("ERROR: No Logitech F310 found (VID=046D PID=C216)")
        print("Make sure the controller is in D mode (not X) and connected via USB")
        sys.exit(1)

    d = hid.device()
    d.open_path(devices[0]["path"])
    d.set_nonblocking(True)
    print(f"Connected: {devices[0].get('product_string', 'unknown')}")
    return d


def read_report(device, timeout_s=5.0):
    """Read until we get a report, up to timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = device.read(64)
        if r:
            return r
        time.sleep(0.02)
    return None


def wait_for_change(device, baseline, timeout_s=10.0):
    """Wait for a report that differs from baseline."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = device.read(64)
        if r and r != baseline:
            return r
        time.sleep(0.02)
    return None


def fmt_report(r):
    return " ".join(f"{b:02X}" for b in r[:8])


def fmt_diff(baseline, report):
    """Show which bytes changed."""
    parts = []
    for i in range(min(len(baseline), len(report), 8)):
        if baseline[i] != report[i]:
            parts.append(f"byte[{i}]: {baseline[i]:02X} -> {report[i]:02X}")
    return ", ".join(parts) if parts else "(no change)"


def test_input(device, baseline, prompt):
    """Ask user to do something, capture the change."""
    print(f"\n>>> {prompt}")
    print("    (waiting for input...)")
    r = wait_for_change(device, baseline)
    if r is None:
        print("    TIMEOUT - no change detected")
        return baseline
    print(f"    Raw: {fmt_report(r)}")
    print(f"    Changed: {fmt_diff(baseline, r)}")
    return r


def test_axis(device, prompt, axis_idx):
    """Ask user to move an axis, show min/max values."""
    print(f"\n>>> {prompt}")
    print("    Move it fully in both directions, then release. (3 seconds)")
    vals = []
    deadline = time.time() + 3.0
    while time.time() < deadline:
        r = device.read(64)
        if r:
            vals.append(r[axis_idx])
        time.sleep(0.02)
    if vals:
        print(f"    byte[{axis_idx}]: min={min(vals)} max={max(vals)} rest={vals[-1]}")
    else:
        print("    No data received")


def main():
    device = open_gamepad()

    # Flush any buffered reports
    while device.read(64):
        pass
    time.sleep(0.2)

    # Get baseline (hands off)
    print("\n=== BASELINE ===")
    print("Take your hands OFF the controller completely")
    input("Press Enter when ready...")
    time.sleep(0.5)
    baseline = None
    for _ in range(10):
        r = device.read(64)
        if r:
            baseline = r
    if baseline is None:
        # Force a report by briefly reading with blocking
        device.set_nonblocking(False)
        baseline = device.read(64, timeout_ms=2000)
        device.set_nonblocking(True)
    if baseline is None:
        print("ERROR: Could not read baseline report")
        sys.exit(1)

    print(f"Baseline: {fmt_report(baseline)}")

    # === FACE BUTTONS ===
    print("\n\n=== FACE BUTTONS ===")
    test_input(device, baseline, "Press and HOLD the A button (bottom)")
    # flush
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press and HOLD the B button (right)")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press and HOLD the X button (left)")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press and HOLD the Y button (top)")
    time.sleep(0.3)
    while device.read(64):
        pass

    # === SHOULDER BUTTONS ===
    print("\n\n=== SHOULDER / TRIGGER BUTTONS ===")
    test_input(device, baseline, "Press and HOLD the LEFT BUMPER (LB)")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press and HOLD the RIGHT BUMPER (RB)")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press and HOLD the LEFT TRIGGER (LT)")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press and HOLD the RIGHT TRIGGER (RT)")
    time.sleep(0.3)
    while device.read(64):
        pass

    # === START / BACK / MODE ===
    print("\n\n=== START / BACK / MODE ===")
    test_input(device, baseline, "Press and HOLD the BACK button (left of center)")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press and HOLD the START button (right of center)")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press and HOLD the MODE button (Logitech logo, center)")
    time.sleep(0.3)
    while device.read(64):
        pass

    # === STICK CLICKS ===
    print("\n\n=== STICK CLICKS ===")
    test_input(device, baseline, "Click the LEFT STICK (press it down)")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Click the RIGHT STICK (press it down)")
    time.sleep(0.3)
    while device.read(64):
        pass

    # === D-PAD ===
    print("\n\n=== D-PAD ===")
    test_input(device, baseline, "Press D-PAD UP")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press D-PAD RIGHT")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press D-PAD DOWN")
    time.sleep(0.3)
    while device.read(64):
        pass

    test_input(device, baseline, "Press D-PAD LEFT")
    time.sleep(0.3)
    while device.read(64):
        pass

    # === STICK AXES ===
    print("\n\n=== STICK AXES ===")
    test_axis(device, "Move LEFT STICK left-right (X axis)", 0)
    time.sleep(0.5)
    while device.read(64):
        pass

    test_axis(device, "Move LEFT STICK up-down (Y axis)", 1)
    time.sleep(0.5)
    while device.read(64):
        pass

    test_axis(device, "Move RIGHT STICK left-right (X axis)", 2)
    time.sleep(0.5)
    while device.read(64):
        pass

    test_axis(device, "Move RIGHT STICK up-down (Y axis)", 3)
    time.sleep(0.5)
    while device.read(64):
        pass

    device.close()
    print("\n\n=== DONE ===")
    print("Mapping complete. Share the output above to verify button assignments.")


if __name__ == "__main__":
    main()
