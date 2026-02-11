#!/usr/bin/env python3
"""Open the MuJoCo viewer with the G1 model (Task 0.3 manual check).

Run this script to verify the native MuJoCo viewer opens and renders
the G1 robot correctly. Mouse orbit/pan/zoom should all work.
Close the window to exit.
"""
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    "reference/unitree_mujoco/unitree_robots/g1/scene.xml")
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
