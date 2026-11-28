# The bussing cell: the bistro's dining room, one Unitree G1 humanoid, the rules

This is the integrator's description of the robot cell your program drives: the site survey, how the lunch shift runs,
the robot and its controllers, and the cart, the tools and the cleaning products as their data sheets describe them.
Everything stated here as exact is exact; everything else (which tables are left, seated or set and when each party
leaves, where the guests' things lie, the soiling, the crumbs, the spill, the robot's start pose, sensor noise) must be
measured in each episode. The cell never reports the pose or the state of anything but the robot itself. There is no
practice cell: section 12 lists what you have to prepare the program with.

## 1. The dining room (exact survey)

Room frame: origin at the south-west floor corner, **x east, y north, z up**, metres, radians, quaternions `w x y z`.
Boxes are given as `[x0, y0, z0, x1, y1, z1]` (on the floor: `[x0, y0, x1, y1]`). `describe()["room"]` holds the same
numbers.

| item | geometry |
|---|---|
| floor, walls | floor z = 0 (chevron oak); inner wall faces at x = 0, x = 10.4, y = 0, y = 8.3; 3.2 m high. The street front is the south wall: four window bays for 0.40-2.30, 2.50-4.40, 4.60-6.50 and 6.70-8.20 in x (sill 0.45, head 2.80) and the glazed entrance door for 8.50 <= x <= 9.50 (2.45 high). The kitchen door (north wall, 6.40 <= x <= 7.20, closed) |
| drawn only (no obstacles) | the street outside the windows, curtains, pictures, wall lights and pendant lamps, the menu board, skirting, the bar shelving's bottles on the station's shelves, table decor (a small vase and a candle on the set tables, drawn only) |
| tables (fixed) | eight tables in three rows; the top at **z = 0.75** (35 mm thick). Two-tops 0.60 x 0.60 on a cast-iron pedestal (column radius 0.04, cross base 0.48 x 0.48, 35 mm high); four-tops 1.00 (x) x 0.75 (y) on four legs 0.05 square, 0.07 in from the corners. Centres: row S (by the windows) T1 (1.25, 0.90), T2 (3.35, 0.90), T3 (5.45, 0.90); row M (four-tops) T4 (2.30, 4.095), T5 (4.80, 4.095); row N (the back) T6 (1.25, 7.29), T7 (3.35, 7.29), T8 (5.45, 7.29) |
| chairs (fixed) | every chair stands south or north of its table, its seat (0.42 x 0.42, top z = 0.46, 30 mm) centred 0.24 m outside the table's edge, its back (0.90 high) on the far side; the guest sitting on it faces the table. Two-tops have one chair each side at the table's x, four-tops two each side at x = centre - 0.24 and + 0.24. Seat ids: `T1S`, `T1N`, ..., `T4Sw`, `T4Se`, `T4Nw`, `T4Ne`, ... (S / N side, w / e of a four-top). A table's west and east ends are free: tables are worked from there |
| a seat's places on the top | its cover (where the guest's plate stood) is centred 0.15 m in from the table's edge in front of the seat; its **roll-up spot** 0.09 m in, at the seat's x, the roll-up's axis along y (square to the edge). `describe()["room"]["tables"]` lists every seat's chair, facing and roll-up spot |
| aisles | between the chair backs of rows S and M (1.65 <= y <= 3.27) and of rows M and N (4.92 <= y <= 6.54), running east-west; the east zone (x > 6.2) is open floor |
| service station (fixed) | against the east wall: a cabinet [9.80, 2.70, 0, 10.40, 4.90, 0.92] with a wooden top at z = 0.92 and open shelves above it (z = 1.35 and 1.75, 0.30 deep, against the wall) |
| roll-up baskets | two wire baskets on the station's top, 0.20 (x) x 0.30 (y), 0.07 high, centred (9.93, 3.15) and (9.93, 3.55); at the start they hold **sixteen clean roll-ups** (lying along y, two layers of four per basket) |
| dish-return pass (fixed) | on the north wall: a stainless counter [7.40, 7.78, 0, 9.40, 8.30, 0.92] under an opening into the dish room (0.92 <= z <= 1.95) |
| cart dock | a yellow taped rectangle on the floor in front of the pass, 7.50 <= x <= 9.30, 6.95 <= y <= 7.74 (section 7) |
| host stand (fixed) | a lectern by the entrance [7.45, 0.55, 0, 7.95, 0.95, 1.12] |
| planters (fixed) | north-west corner: pot radius 0.22 at (0.40, 7.95), 0.55 high, foliage radius 0.42 up to z = 2.05; by the south-east window: pot radius 0.20 at (8.85, 0.35), 0.45 high, foliage radius 0.40 up to z = 1.70 |
| the mop corner | on the east wall, north of the station: the **mop's clip** with its jaw centre at (10.30, 5.45, 1.10); the mop hangs in it, its handle upright, its head on the floor along the wall; the **mop bucket** (a yellow wringer bucket, fixed) [9.95, 5.84, 0, 10.35, 6.12, 0.42] |
| the cone's spot | a taped square on the floor by the east wall, centred (10.05, 6.58): the yellow wet-floor cone stands there at the start |
| the bus cart's home | by the station: the cart at (8.05, 3.80), heading pi (its front to the west, its push handle to the east) |

## 2. The shift (what varies, what happens)

* **The tables.** At the start each table is **dirty** (its party has just left), **occupied** (a party is finishing
  lunch and leaves at some time during the shift) or **set** (one roll-up at every seat, nothing to do): one or two
  are dirty, three or four occupied, the rest set; which is which, and when each party leaves (between about 4 and
  40 minutes into the shift, at least 200 s apart), varies. A party leaves all at once (they get up and go; nothing of
  theirs stays on the chairs).
* **What a party leaves.** At every seat a plate (often with food left and the fork and the knife on it or beside it), a
  glass (with some of the drink left in it), a fork, a knife and a napkin (crumpled, on the plate or the top); crumbs
  round the covers; and soiling of the top: food smears by the plates and drink rings under the glasses. Where exactly
  everything lies varies. A seated party's things are theirs: do not take them (the instruction's scoring).
* **The spill.** Once in the shift (between about 5 and 30 minutes in), a guest at an occupied table knocks a drink
  over: a puddle of 0.3 to 0.9 m across on the floor of the aisle beside their table (cola, orange juice or red wine),
  while the party is still seated. It stays wet until it is mopped up (section 9).
* **The dish crew** works in the dish room behind the pass (section 7, the dock).
* **The robot** starts near (9.05, 2.95), facing west, beside the cart's push handle (within 8 cm in x and y and
  0.15 rad of heading); the cart stands at its home with its tub empty, the cloth in the sanitizer bucket and the crumb pan on its
  hook.
* **Sensor noise** (section 6) and the camera images' white balance vary per episode.

## 3. The robot

One Unitree G1 humanoid (29 DoF with Dex3-1 hands). The model is MuJoCo Menagerie's `unitree_g1/g1_with_hands.xml`
(copy in `~/given/robot/unitree_g1/`, BSD-3-Clause) with the cell's modifications (those of the house-cleaning cell of
this benchmark):

* **Locomotion controller.** The pelvis rides four base joints: `base_x`, `base_y` (room x / y), `base_yaw` and
  `base_z`, driven by force-limited servos (x / y 300 N, yaw 150 N m) that the robot's own locomotion controller
  commands; the legs are posed from the stand height and do not touch the floor (no balance falls are simulated). A
  robot obstructed by furniture stops; its controller does not wind up beyond 5 cm / 0.1 rad, but while it stays
  obstructed the base keeps pushing against the obstacle with up to its servo limit (300 N): held for 0.2 s, that is
  a safety violation (section 10). Cancel the goal (or stop the velocity) when the base stops short.
* **Legs in the camera images.** The cameras draw the legs stepping while the base travels or turns (drawn only: the
  physics keeps the posed legs; nothing above the hips changes in the images).
* **Stand height** (pelvis height above the floor) 0.48 - 0.80 m, nominal 0.793 m; lower means a squat (the knees go
  forward: about 0.21 m ahead of the pelvis at 0.70, 0.34 m at 0.48, at knee height). The waist pitches forward up to
  0.52 rad.
* Every robot body is gravity-compensated; payloads are not.
* Joints: waist yaw / roll / pitch, 7 per arm (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw), 7 per hand;
  ranges as in the model (`describe()` lists them).
* **Collision:** body links are capsules (Menagerie's `g1_mjlab.xml` colliders); the robot collides with the furniture,
  the cart, the loose objects and the guests, never with itself or the floor. **The hands (the `wrist_yaw_link` and the
  finger links) carry no contact**: grasping is the cell's grip rule (section 4). **What a hand holds does not collide
  with the robot either** (it is part of the hand while held); it collides with everything else, the guests included.
* **Grasp site** of each hand, `<side>_grasp`: in the `<side>_wrist_yaw_link` frame at (0.150, -0.046, 0) for the left
  hand and (0.150, +0.046, 0) for the right hand, with the link's axes: **x = approach** (along the fingers), **y = pinch
  axis**, z = x cross y.
* **Head camera** `head`: an RGB-D camera on `torso_link` at (0.0576, 0.0175, 0.4299), optical axis along the link's +x
  pitched down by 0.8308 rad; it moves with the waist.

## 4. Grip rule (hands)

Closing a hand (`"close"`) engages a grip on an object when, at that instant, the hand's grasp site lies at one of the
object's grasp features:

* **bars and cylinders** (glasses, the forks' and knives' handles, roll-ups, the cart's push handle, the crumb pan's
  handle, the cone's carry bar, the mop's handle): the grasp site within 15 mm of the surface (at most the radius +
  15 mm from the axis), between the feature's ends (glasses from 12 mm above their foot to 8 mm below their rim; a
  fork's or a knife's handle, its last 8 cm (the end away from the tines or the blade); a roll-up from 2 cm inside
  either end; the cart's handle from 3 cm inside its ends; the crumb pan's handle from 3 cm above its root to its end;
  the cone's bar from 1 cm inside its ends; the mop's handle from 0.35 m above its swivel to 1 cm below its top), the
  pinch axis within 35 deg of perpendicular to the axis;
* **slabs** (the cloth, which counts as a slab both across its thickness and across its width): the pinch axis within
  35 deg of the slab's thickness or width, the grasp site within 15 mm of the slab and at most 45 mm inside one of its
  edges or faces (and at least 5 mm from the corners along it);
* **discs** (plates): the pinch axis within 35 deg of the plate's axis, the grasp site within 15 mm of its mid-plane and
  at most 45 mm inside its rim;
* **lumps** (crumpled napkins): the grasp site within 12 mm of the napkin's bulk (42 mm from its middle).

The fingers close on what the hand takes hold of and stop where they meet it or what it rests on (into the cloth or a
napkin they sink a few millimetres); a hand that closes on nothing closes fully. The fingers' joint angles
(`get_state()`) show the hand's shape. A loose object is then held rigidly at the relative pose of that instant until
the hand opens or the grip slips; what rests on a held plate (its cutlery, a napkin) rides on it as anything resting on
something does. A closed hand that found nothing holds nothing (open and close again). Grip limits: an object held by
one hand slips when the hand's load exceeds **30 N** or its torque about the grasp site exceeds **5 N m** (**20 N m**
when the hand holds it by a bar or cylinder feature: a power grasp); held by both hands, when its net load per hand
exceeds 30 N; an excess must last 50 ms. The hand's tactile reading (`get_state()`: `tactile_N`) is its share of the
held object's load (0 when it holds nothing). **The cart's push handle** is a bar of the cart: a hand gripping it holds
the cart, which then follows the hands (section 7).

## 5. Control (each 10 ms tick)

* Base: `base_velocity` [vx, vy, wz] in the robot's heading frame (x forward, y left; limits 0.5, 0.3 m/s, 0.8 rad/s;
  accelerations 0.8, 0.8 m/s^2, 1.5 rad/s^2), held until replaced; or `base_goal` [x, y, yaw] in the room: a straight-line
  approach (proportional, gain 1.2 /s, within the same limits, no obstacle avoidance) that stops within 5 mm / 0.01 rad.
  `stand_height` moves at 0.15 m/s.
* Waist: joint targets [yaw, roll, pitch], at most 0.8 rad/s.
* Arms: joint targets (`q`, 7 values), or a grasp-site pose target (`pose` [x, y, z, qw, qx, qy, qz]) or position target
  (`pos` [x, y, z]: the wrist's orientation left free) in the robot's **base frame** (origin on the floor under the
  pelvis, x along the heading, z up; the default) or in the **room frame**. A target is solved by the cell's
  damped-least-squares IK over the arm's 7 joints (base, waist and legs as they are): a target within 0.12 m / 0.6 rad of
  the arm's previous target keeps the arm's configuration, a farther one gets a seeded global solve. `path: "line"`
  (default) moves the grasp site on a straight line with a minimum-jerk timing (about 0.25 m/s and 1 rad/s on average,
  peak accelerations at most 1 m/s^2 and 4 rad/s^2); `path: "joint"` moves in joint space to the target's IK solution.
  Joint targets move at most 1.5 rad/s with 6 rad/s^2. An unreachable target leaves the arm where the IK gets closest
  (`get_state()`: `ik_residual` [position m, rotation rad]). A target in the base frame travels with the base.
* Hands: `"open"` or `"close"` (closing applies the grip rule once).

## 6. Sensors

* `get_state()`: the localised base pose [x, y, yaw] (independent noise of about 3 mm and 0.2 deg per reading), the
  stand height, whether the base is moving, the motor state of the waist, the arms and the hands (`q`, `dq`,
  `tau_est`, below), per hand `closed`, `tactile_N`, the grasp site's pose in the robot's base frame (exact kinematics)
  and the IK residual of a pose target.
* **Motor state** (`q`, `dq`, `tau_est`: one {joint name: value} map each), as the G1's low-level state reports it
  (the body's `LowState` and the Dex3-1 hands' `HandState`, one entry per motor), for every motor a program can read:
  the waist's 3, the arms' 2 x 7 and the hands' 2 x 7 (`describe()`: `robot.motor_state` lists them with their torque
  limits). The legs belong to the locomotion controller (section 5) and are not reported.

  | field | unit | what it is | accuracy |
  |---|---|---|---|
  | `q` | rad | the joint angle, from the joint's encoder | exact |
  | `dq` | rad/s | the joint velocity | reading-to-reading noise of about 0.01 rad/s (1 sigma), quantised to 0.001 rad/s |
  | `tau_est` | N m (the joint's sign convention) | the joint torque estimated from the motor's current | a calibration offset of up to a few percent of the joint's torque limit (1.5 %, 1 sigma) and a scale error of up to 5 %, both constant through an episode (read them at rest), plus reading-to-reading noise of about 0.5 % of the limit, quantised to 0.01 N m |

  `tau_est` is what the drive delivers: the torque holding the robot's own links up (the gravity feed-forward), the
  payload, the arm's acceleration, and the load of anything the arm or the torso presses on or is pressed by (a contact
  far from the joint's axis shows; a push along a link straight through a joint does not). A hand's fingers show their
  drives' own effort only: holding is the grip rule (section 4), so what a hand holds shows in its `tactile_N` and in
  the arm's and the waist's torques, not in its fingers'.
* Cameras (`get_camera`): `ceiling_sw`, `ceiling_nw`, `ceiling_center` and `ceiling_service`, fixed RGB-D cameras
  960 x 720 (vertical field of view 72 deg) whose exact poses and intrinsics `describe()` gives (between them they see
  the whole dining room; `ceiling_service` looks over the east zone: the station, the cart's home, the mop corner and
  the dock); `head`, 640 x 480 (58 deg), whose pose in the room at capture time comes with each frame (composed from
  the localised base pose and the exact kinematics). Pinhole, no distortion: fx = fy = (H / 2) / tan(fovy / 2),
  principal point at the image centre. Depth is along the optical axis, registered to the colour pixels, 1 mm
  quantisation, noise sigma 1.5 mm + 2.5 mm/m^2 z^2, invalid (0) below 0.28 m (head) / 0.40 m (ceiling), beyond 6 m,
  on depth discontinuities and on 2 % of the pixels at random. Colour is rendered physically based
  (MuJoCo's Filament renderer): the midday light through the south windows, the pendant lamps over the tables, the
  wall lights, the light the room reflects; a per-episode white balance / brightness draw and pixel noise. The guests,
  the tableware, the food, the soiling and the crumbs, the spill (and how wet the floor still is), the cloth's colour
  (it darkens as it takes up soil) are drawn as they are. **At most 1500 frames per episode** over all cameras.

## 7. The bus cart and the tools

* **The bus cart.** A two-shelf stainless cart on four casters (mass 15 kg). Its frame: origin on the floor under the
  middle of its top shelf, **x towards its front** (away from the push handle), y to its left, z up; its footprint
  0.95 (x) x 0.40 (y). Top shelf at z = 0.76 with a 22 mm lip round it, lower shelf at z = 0.30; the **push handle** is a
  horizontal bar (radius 12.5 mm) along y at x = -0.545, z = 0.98, 0.34 long, on two posts. At home the cart stands at
  (8.05, 3.80) heading pi. **Pushing:** while one or both hands grip the handle (the grip rule, section 4), the cell
  drives the cart in the plane (its two slides and its turn; its height and tilt are fixed) towards the pose the holding
  hands imply: each hand's grasp point stays where it was on the handle when the hand closed (with two hands their
  midpoint and the line between them set the cart's turn as well), with a servo whose force per holding hand is the
  hands' load (30 N per hand, 50 ms over: the handle slips out of the hands); a hand whose grasp site moves more than
  6 cm off the handle lets go of it. The cart collides with the furniture, the loose objects and the guests while it is
  held. Let go of, its casters' brakes hold it where it stands.
* **The bus tub** stands on the top shelf at the front (its middle at cart (0.18, 0)): a grey polypropylene bus box,
  outer 0.58 (x) x 0.36 (y), 0.14 high, walls 6 mm; inside it spans cart x -0.104 to 0.464, y -0.174 to 0.174, its floor
  at z = 0.766 and its rim at z = 0.90. Things put into it ride with the cart.
* **The sanitizer bucket** stands on the top shelf at the back (its axis at cart (-0.30, 0)): a red pail, inner radius
  0.115, 0.10 high (its rim at z = 0.86), holding sanitizing solution 88 mm deep; a perforated rack inside at 70 mm above
  its bottom, under the solution's surface, on which the cloth lies soaking at the start.
* **The cloth** (a folded red bar towel with two white stripes, 0.13 x 0.09 x 0.018, 70 g): a slab for the grip rule
  both across its thickness and across its width (held from above across its width, it hangs flat under the hand). It
  gives under pressure: its pressed face sinks into the fold against a spring of 0.6 N per mm, by up to 10 mm (then it
  is pressed solid). A damp cloth slides easily on the varnished tops (friction coefficient 0.35).
* **The crumb pan** (a black table crumb pan, 0.22 wide, 0.16 deep from its open lip to its back wall, walls 0.05 high,
  0.22 kg; its handle, radius 11 mm, 0.13 long, rises from the back wall tilted 28 deg back) hangs on a hook below the
  cart's push handle (the hook at cart (-0.505, 0, 0.74)): its handle let go of within 20 mm of the hook, the pan
  hanging lip down within 20 deg of the vertical, hangs on it; a hand that grips it takes it off.
* **The wet-floor cone** (yellow, 0.82 high on a 0.30 x 0.30 base, 0.75 kg): a carry bar (radius 12.5 mm, 0.11 long, at
  its top, horizontal) to lift it by; it stands on its spot at the start (its bar along y).
* **The flat mop.** An aluminium handle (radius 12.5 mm, 1.40 long, a blue grip at its top), its frame's origin on the
  head's swivel, the handle along its z axis; the head (0.40 x 0.11 frame with a microfibre pad 0.42 x 0.13 under it, the
  pad's underside 46 mm below the swivel) hangs on a two-axis swivel (+-65 deg each way, damped), so it stays flat on
  the floor while the handle leans; the head's long side runs along the handle frame's x axis. The pad is mounted on a
  sprung plate: it gives against a spring of 0.5 N per mm by up to 20 mm either way, so the head follows the floor
  under a hand that holds the handle a little low or high. Handle 0.35 kg, head 0.40 kg. A wet microfibre pad glides on
  the floor (friction coefficient 0.15).
* **Clips.** The mop's clip on the east wall: a mop let go of with its handle's axis within 12 mm of the jaw centre
  (10.30, 5.45, 1.10) and the handle within 12 deg of vertical clicks in and hangs there; a hand that grips a clipped
  mop takes it out of the clip.
* **The dock and the dish crew.** A cart left standing in the dock (its whole footprint inside the taped box, its long
  side along the pass, i.e. heading 0 or pi, within 12 deg, no hand on it) for 20 s is served once: the dish crew takes
  everything that lies in the tub (tableware, linen, roll-ups) and the crumbs in it, and changes the sanitizer (the
  cloth, if it lies in the bucket, comes back wet with fresh sanitizer). The crew serves a cart again only after it
  has left the dock.

## 8. The tableware

* **Plates** (white porcelain, radius 0.12, 22 mm high, 0.42 kg): a disc for the grip rule. **Glasses** (tumblers,
  radius 35 mm, 0.115 high, 0.22 kg with the drink left in them). **Forks** (0.20 long) and **knives** (0.215 long):
  held by their handle's end. **Napkins**: crumpled linen lumps. **Roll-ups**: a fork and a knife rolled in a napkin,
  banded (0.23 long, radius 21 mm, 0.11 kg); they lie on a top as they are put (they do not roll).
* **What breaks.** A plate or a glass survives a fall onto a hard surface of up to about 30 cm (an impact of up to
  2.5 m/s) and a steady load of up to 600 N (a plate) or 400 N (a glass); a harder impact or a heavier load breaks it.
  A broken dish earns nothing, and the restaurant charges for it (the instruction's scoring).

## 9. The cleaning (what the cloth, the sanitizer, the pan and the mop do; never reported)

The cell simulates the cleaning the way the products' and the tools' data sheets describe them; their states are never
reported, only drawn in the camera images.

* **Soil.** The food smears and drink rings on a top come off under the cloth only while it lies flat on the top (its
  face within 25 deg of it), pressed on it and sliding over it: it starts to work at half a newton; from about 4 N on,
  harder pressing adds nothing. One firm pass takes most of a smear off. A cloth wet with sanitizer works best, a merely
  damp one about a third as well, a dry one hardly at all. What comes off loads the cloth: a loaded cloth takes less,
  and one near its load capacity (about one and a half tables' worth of smears) smears soil back where it passes.
* **Sanitizer.** Lying in the bucket's solution for a second wets the cloth with sanitizer and rinses its load out.
  Wiping uses the sanitizer up: one dip wets about 1.6 m^2 of top. A top counts as sanitized where the cloth passed
  over it pressed while it still held sanitizer (in 5 cm squares), after the party left.
* **Crumbs** rest on the highest surface below them and move only when pushed. The cloth pressed on a top pushes the
  crumbs under it ahead of it; a crumb pushed over an edge flies on a few centimetres the way it was pushed and falls:
  onto the floor, a chair's seat, the crumb pan if its opening is there (the pan within 35 deg of level), or into the
  tub. Crumbs in the pan ride with it and pour out over its lowest edge when it is tipped past 55 deg; tipped over the
  tub, they fall into it. Crumbs in the tub (or on the cart's top shelf) ride with the cart; the dish crew empties the
  tub.
* **The spill and the mop.** The spill is a film of drink on the floor, drawn as it is (darker and glossier where more
  is left). The mop's pad takes it up only while it lies flat on the floor (within 20 deg of level, its underside within
  12 mm of the floor), pressed on it (more than 3 N) and moving over it: each pass of the pad over a spot takes most of
  what is left there; the pad holds far more than one spill. Where less than a twentieth of a spot's drink is left, the
  spot is damp and dries 90 s later; the floor is **dry** when every spot of the spill has dried.
* **The guests.** A seated party's things (their plates, glasses, cutlery and napkins) are theirs until the party
  leaves; a hand that takes one is charged (the instruction's scoring). Touching a guest is a safety violation
  (section 10).

## 10. Safety (a violation makes the episode score 0; the episode goes on)

* a robot link other than the hands pressing on anything with more than 150 N for 0.2 s;
* a loose object let go of (or slipping) while moving faster than 1.5 m/s (thrown);
* a loose object not held, and not let go of in the last 2 s, moving faster than 4 m/s (struck);
* **anything touching a guest** (the robot, what it holds, the cart it pushes) with more than 5 N for 30 ms.

## 11. Budgets per episode

4200 s of simulated time (the shift; the episode ends there); 3600 s of program time (the episode's wall clock minus
the cell's own time inside `step`, simulating, and inside `get_camera`, rendering: your program's computation and all
its other requests count); 120 s from a reply to the next request (silence) ends the episode; 1500 camera frames; at
most 500 ticks per `step` call. The program's process (all its threads) may use at most 14,400 s of CPU time per
episode (4 x the program-time budget: the operating system ends it with SIGXCPU beyond that, and the episode is scored
as it stands); its numerical libraries run with OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS set to 2 (a
thread pool of your own, e.g. OpenCV's, counts against the CPU budget too: `cv2.setNumThreads` keeps it small).

## 12. Preparing the program (no practice cell)

The program is deployed at the restaurant as it is: nothing in your session runs the cell. You have:

* `describe.json` (this folder): exactly what the cell's `describe()` returns (the robot, joint names and ranges,
  control limits, grip limits, the cameras' intrinsics and the ceiling cameras' exact poses, the head camera's
  mount, the room's survey, the budgets, the safety limits);
* `survey/` (this folder): survey photos of two shifts of the same bistro (`shift_a`, `shift_b`), drawn by the rules of
  section 2 but not among the graded ones: per shift the four ceiling cameras at the start and the two that see the
  spill best 15 s after it, and a head-camera frame from the west end of a table (0.30 m from its edge, facing it);
  `shift_a` also head-camera frames before the station's roll-up baskets and before the mop corner (`<name>.json` holds
  the frame's metadata as `get_camera` returns it, `<name>_rgb.png` and `<name>_depth_mm.png` its colour and 16-bit
  depth; `bistro_client.client.load_frame(<name>.json)` gives the reply that `decode_frame` reads; `index.json` says
  what each frame shows and when it was taken);
* the robot's MuJoCo model (`~/given/robot/unitree_g1/`), from which the cell's robot is built (section 3), for your
  own kinematics, reach and collision checks.
