"""Real-time IGRIS-C simulation server for PICO 4 Ultra teleoperation / data collection.

PICO tracking (XRoboToolkit stream or the PICO app over UDP) -> streaming correction per tracker
profile -> streaming retargeting (the igris_pipeline mink IK) -> MuJoCo scene with interactive
objects -> state back to the headset app, and episodes recorded in the igris_pipeline format.

    scene.py      scene JSON (shared with the Unity app) -> MuJoCo model with the robot
    streaming.py  causal correction + retargeting, calibrated on a short still window
    world.py      MuJoCo physics: kinematic / harness robot, grasping, object states
    session.py    one teleoperation session: calibration, locomotion, recording, state messages
    protocol.py   UDP JSON protocol (docs/sim_protocol.md) and the scene asset HTTP server
    sources.py    tracking sources: XRoboToolkit, simulated, replay of a recorded episode
"""
