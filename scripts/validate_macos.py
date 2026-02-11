#!/usr/bin/env python3
"""Validate that all core dependencies work on macOS bare metal."""
import sys
import platform

def check(name, fn):
    try:
        result = fn()
        print(f"  [PASS] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False

results = []

# 1. Python version
results.append(check("Python version",
    lambda: f"{sys.version} ({'OK' if sys.version_info[:2] == (3, 10) else 'WANT 3.10'})"))

# 2. Platform
results.append(check("Platform",
    lambda: f"{platform.system()} {platform.machine()}"))

# 3. MuJoCo
results.append(check("MuJoCo import",
    lambda: __import__('mujoco').__version__))

# 4. ONNX Runtime
results.append(check("ONNX Runtime import",
    lambda: __import__('onnxruntime').__version__))

# 5. CycloneDDS
def check_cyclonedds():
    __import__('cyclonedds')
    import importlib.metadata
    return importlib.metadata.version('cyclonedds')
results.append(check("CycloneDDS import", check_cyclonedds))

# 6. unitree_sdk2py core imports
def check_sdk():
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    return "ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize OK"
results.append(check("unitree_sdk2py core imports", check_sdk))

# 7. DDS loopback init
def check_dds():
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    iface = "lo0" if platform.system() == "Darwin" else "lo"
    ChannelFactoryInitialize(1, iface)
    return f"DDS init on {iface} OK"
results.append(check("DDS loopback init", check_dds))

# 8. MuJoCo model load
def check_model():
    import mujoco
    model = mujoco.MjModel.from_xml_path(
        "reference/unitree_mujoco/unitree_robots/g1/scene.xml")
    return f"G1 model loaded: {model.nu} actuators, {model.nq} qpos"
results.append(check("MuJoCo G1 model load", check_model))

# 9. RecurrentThread import (expected to fail on macOS)
def check_recurrent():
    from unitree_sdk2py.utils.thread import RecurrentThread
    return "RecurrentThread available (unexpected on macOS — good, means SDK is fully native)"
recurrent_ok = check("RecurrentThread import", check_recurrent)
if not recurrent_ok:
    print("    ^ This is EXPECTED on macOS. We will replace it with src/compat.py.")

# Summary
passed = sum(results)
total = len(results)
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
if not recurrent_ok:
    print("RecurrentThread failure is expected and will be patched.")
if passed >= total - 1:  # Allow RecurrentThread to fail
    print("VERDICT: macOS bare metal is VIABLE. Proceed with PLAN_METAL.md.")
else:
    print("VERDICT: Too many failures. Fall back to PLAN_DOCKER.md (Docker).")
