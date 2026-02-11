**Project documentation files (`SPEC.md`, `PLAN_DOCKER.md`, `PLAN_METAL.md`, `WORK.md`) must never be deleted.** They are the authoritative record of requirements, design decisions, and implementation plans. They should be updated in place as the project evolves.

---

your mission - should you choose to accept it - is to help me build an all in one deployment stack for the unitree g1 robot!

the deployment stack must meet these minimum requirements:
- runs in docker as ubuntu 20.04
- can be run on both mac and linux machines (iOS, ubuntu 20, 22, 24)
- uses viser for visualization (because mac cannot x11 forward the mujoco viewer, we want to use mujoco in headless mode when on mac and do all visualization in viser instead)
- the viser visualization should be interactive. the simulation/policy starts and stops based on buttons in viser. the sim can be reset from a button. the policy can be switched from a drop down, etc. big red button for e stop to put the robot in damp mode.
- can evaluate RL policies both in simulation and on the real robot
- maximally shares code between sim and real workflows
- uses the unitree sdk

There are many example codebases. the simplest that we can base our work on is the unitree_mujoco repo using the python workflow from unitree itself. this repo is currently in the references dir while we build our version. we will not edit the files in the reference dir, but we may re implement any part of it in our repo as we see fit. please see these references to learn how the unitree sdk is used:

uses cpp but has nice ros interface and docker:
https://github.com/HybridRobotics/motion_tracking_controller

ros based setup
https://github.com/catachiii/crl-humanoid-ros

python interface for the sdk
https://github.com/amazon-far/holosoma

for the boston dynamics spot, but has a nice interface:
https://github.com/boston-dynamics/spot-rl-example

you can also see the unitree github repositories, for example the unitree sdk and ros packages:
https://github.com/unitreerobotics
https://github.com/unitreerobotics/unitree_sdk2
https://github.com/unitreerobotics/unitree_sdk2_python
https://github.com/unitreerobotics/unitree_ros
https://github.com/unitreerobotics/unitree_ros2

please consult the viser source code and docs:

https://github.com/nerfstudio-project/viser
https://viser.studio/main/

and see how mjlab uses viser as an interactive viewer:

https://github.com/mujocolab/mjlab


Question answers:
1) we will target both the 23 dof and 29 dof G1, with the 29 dof version being default. we do not have dex hands. I do not know about the IDL modes, please explain to me the difference.
2) the policies are exported onnx policies. see IsaacLab for the format. We will need to format observations for the joint pos, joint vel, last action, lin vel, ang vel, projected gravity, velocity command, etc. the action space is joint pos offsets for the PD torque law tau = Kp(q_home + Ka a - q) - Kd qdot. The policy runs at 50 hz, the sim should run at 200 hz. this should be configurale. There should be a class to handle evaluation of the policy, yes.
3) yes the browser should be accessible anywhere on the local network (but obviously not the broader internet). mutliple camera angles should be supported, woth the default being to follow the robot. this should be interactive in the viser scene. the camera view should be changable with either a keyboard input or a drop down menu. visualization of the mesh and the collision geometries should both be suppoted and swappable based on either keyboard input and/or a button. the terrain can assumed to be flat for now (does not need to be visualized explicitly)
4) e stop should be latching. for now only one safety mode should be supported (damping). e stop should occur in both sim and real. damping mode means that the desired position is the current position (zero contribution to torque) so the only toque applied (softly) opposes the direction of motion. 
5) there should be different commands to run from the command line for sim vs real. for example, we provide the specific interface name to connect to the robot on when attached via ethernet. in the future we may also support wifi, but for now only connecting the computer to the robot via ethernet is supported. there should be a unified interface with subclasses for sim and real. 
6) sim and real can be separate containers or the same container. whatever is best. viser connects to a specific port, see the viser docs for details. we do not care which port it uses. use EGL for headless mujoco rendering (assuming that works on both mac and linux). GPU policy acceleration is not needed for now, as the policies are small. the volume strategy should be such that code changes to this repo inside or outside of the container are reflected in both places. logs policies and configs should all be mounted that way as well.
7) no, mujoco is fast to launch so no dev mode is needed for now. logging should be easy to parse and play back, and ideally compressed so the logs dont get too big. rosbags are not ideal since it requires launching the container to unpack from a mac. neither tensorboard nor wandb support are needed at this time, but we should have a way to log eg observations, actions, state, etc.
8) newer python would be great as long as it plays nice with all the deos, mujoco, viser, onnxrutime, ros, etc. we dont want a managed python hell, so whatever is simplest in the container. no specific versions are needed to my knowledge. the engineer should explore both ros/ros2 solutions and pure python sdk solutions and provide justification for the choice.
9) i meant macos, not ios. and yes, pure mj_step is the intention (no mujoco viewer window).

please write a full task spec and save it to an md file accordingly. thanks!