# NOTICE (grader-side; never staged into the agent's home)

The robot is the MuJoCo Menagerie `unitree_g1` model (`g1_with_hands.xml`; BSD-3-Clause, its LICENSE ships inside the
folder), fetched by `environment/Dockerfile` at the commit pinned file by file in `environment/g1_assets.sha256`. The
cell replaces the model's floating base and actuators (base joints, servos, gravity compensation) and uses the primitive
colliders of Menagerie's `g1_mjlab.xml`; `tests/bistro_cell/robot.py` builds it with MjSpec at cell start, as
house_cleaning_with_tools's cell does (the same modifications). The agent gets the unmodified folder in
`~/given/robot/unitree_g1/`.

The cameras' renderer: `bistro_cell/render.py` (colour through MuJoCo's Filament module, Apache-2.0, part of the
mujoco wheel, on Mesa's lavapipe; depth through OSMesa; the RealSense-like sensor model), house_cleaning_with_tools's
renderer with the bistro's lights, a procedural image light (`bistro_cell/ibl.py`: an equirectangular sky and room
probe made in code, prefiltered into the KTX cube map and spherical harmonics Filament reads) and `bistro_cell/look.json`
(the lights' and the materials' terms); the robot's legs as the cameras draw them while it moves
(`bistro_cell/walk_core.py`, a byte copy of house_cleaning_with_tools's: footsteps planned from the base motion, leg
inverse kinematics on the G1 model's own leg chain; no motion-capture or third-party animation data).

Everything in the dining room is procedural and original to this task: the room's shell, the windows and the street
outside, the chevron floor, the tables, chairs, the service station and its baskets, the dish-return pass, the host
stand, the planters, the lamps and the decor, the bus cart with its tub, sanitizer bucket and crumb pan, the mop and its
bucket, the wet-floor cone, the tableware, the food and the drinks, the roll-ups, the cloth, the guests (simple seated
figures), their meshes and textures made in code (`bistro_cell/props.py`, `bistro_cell/meshes.py`,
`bistro_cell/textures.py`, `bistro_cell/objects.py`, `bistro_cell/scene.py`) and their colliders primitive shapes; so
are the cleaning processes (`bistro_cell/cleaning.py`: the soil and the sanitizer, the cloth's load, the crumbs, the
spill, the mop and the drying), the shift (`bistro_cell/service.py`: the cart's servo, the guests leaving, the spill,
the dish crew), the layout draw and the scoring. No third-party model, texture or dataset is part of the room, the
physics, the cleaning objects or the score; there is no asset download.
