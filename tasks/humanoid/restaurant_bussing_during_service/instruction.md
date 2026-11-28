# Bus a bistro's dining room through a lunch shift with a Unitree G1 humanoid

You are the robot integrator of a restaurant group. Its bistro has hired a busser for the lunch shift: one Unitree G1
humanoid (29 DoF, Dex3-1 hands), standing by the bus cart at the service station when the shift starts. Your job:
write the program that makes the robot do the busser's work for the whole shift, using only what the robot can sense:
its head camera, four ceiling cameras of the dining room, its motors' state (joint angles, velocities and torque
estimates), its localisation and its hands' tactile load. The dining room, its furniture, the bus cart and the tools are
surveyed exactly (`~/given/cell/cell_manual.md`); nothing else is given: which tables have been left, which are still
seated and when each party leaves, where their plates, glasses, cutlery and napkins lie, how soiled the tops are, where
the crumbs fell, when and where a guest spills a drink — all of that has to be seen in the camera images (and
remembered). The room changes while the robot works: parties finish and leave during the shift, so the program has to
keep watching the room, not only plan once. Every shift is different, and there is no practice cell: the program goes
to the restaurant as it is, the way a busser robot does.

## The busser's job (the floor manager's checklist)

1. **Turn the tables.** A table whose party has left (some are left from the start; the seated parties leave during the
   shift) is turned in three steps:
   * **cleared**: every seat's plate, glass, fork, knife and napkin into the bus tub on the cart's top shelf (a plate
     may go in with its cutlery and napkin still on it), unbroken — a dish dropped onto something hard, or crushed,
     breaks;
   * **wiped and sanitized**: the top wiped with the bar towel from the sanitizer bucket on the cart (press it flat on
     the top and slide it; a dry or used-up cloth does little, dip it again) — the food smears and the drink rings come
     off, and the top counts as sanitized where the cloth passed pressed while it still held sanitizer; the crumbs on
     the top go into the crumb pan (on its hook below the cart's push handle) or the tub, not onto the floor;
   * **reset**: a clean roll-up (cutlery rolled in a napkin, from the two baskets on the service station's counter) laid
     at every seat, in front of it and square to the table's edge.
   **Never touch a seated party or take their things**: a table is worked only once its party has left.
2. **The spill.** At some moment a guest spills a drink on the floor of the aisle beside their table. Set the yellow
   wet-floor cone (on its spot by the east wall) up beside the spill **at once**, mop the spill up with the flat mop
   (hanging in its clip on the east wall), hang the mop back in its clip, and once the floor is dry put the cone back
   on its spot.
3. **The dish return.** The dish crew takes the tub's load when the cart is left standing in the dock (the yellow taped
   box in front of the dish-return pass on the north wall): bring the cart there as the tub fills, and at the end.
4. **Closing.** When the shift ends: the cart back at its home by the service station, the cloth in the sanitizer
   bucket, the crumb pan on its hook.

The cell manual gives the dining room's survey, the robot, its controllers and the grip rule exactly, the bus cart, the
tub, the bucket, the cloth, the crumb pan, the mop, the cone and the clips, how the cleaning works (wiping, the
sanitizer, the crumbs, mopping, drying) the way the products' and the tools' data sheets describe it, how the shift
runs (the parties leaving, the spill, the dish crew), what breaks, safety and the budgets.

## What you produce (in `/home/user/submission/`)

`busser.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once per
hidden shift (**3 shifts**) in its own physics cell; the program talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`). Per shift: 4200 s of simulated time (the lunch shift: the episode ends there); 3600 s of
program time, which is the episode's wall clock minus the time the cell spends simulating (inside `step`) and rendering
(inside `get_camera`), so it measures your program's own computation and its other requests; at most 120 s from a reply
to your next request; 1500 camera frames in total; at most 14,400 s of CPU time for the program's process (all
threads; SIGXCPU ends it beyond that), whose numerical libraries run with `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` /
`MKL_NUM_THREADS` = 2.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable HOME / TMPDIR, a
minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL`, `PYTHONPATH` = a copy of `~/given/lib`,
`ARB_GIVEN_DIR` = a read-only copy of `~/given`) and the launcher `python3 -B -E -s -c <bootstrap> <lib> busser`
(or `bash run.sh`). Resolve given files through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`, write
only under the run's HOME / TMPDIR. A program that crashes ends that shift with what it achieved; the other shifts
still run. Your submission must be final when your session ends: the submission folder is then copied out of your
sandbox and graded in a fresh sandbox that holds nothing else of yours (no other file, no process you started).

## What you are given

| path | content |
|---|---|
| `~/given/cell/cell_manual.md` | the cell: the dining room's survey (exact), the shift's rules, the robot and its controllers, the grip rule, the sensors and cameras, the cart and the tools, the cleaning rules, what breaks, safety, budgets |
| `~/given/cell/cell_api.md` | the cell's programming interface |
| `~/given/cell/describe.json` | exactly what the cell's `describe()` returns (robot, joints, limits, cameras with their exact poses, the room's survey, budgets, safety) |
| `~/given/cell/survey/` | survey photos: camera frames of two shifts of the same bistro drawn by the same rules as the graded ones (not among them), as the cameras deliver them (cell_manual.md section 12) |
| `~/given/robot/unitree_g1/` | the robot's MuJoCo model (Menagerie, BSD-3-Clause); the cell's modifications are in the manual |
| `~/given/lib/bistro_client/` | the cell's client (`Cell`, `decode_frame`, `load_frame` for the survey photos, the launcher) |

**No practice cell.** Nothing in your session runs the cell: no practice episodes, no answer key, no checker. You
prepare the program from the manual, the survey, the photos and the robot's model (you may build any test rig of
your own); it then runs on the graded shifts as it is.

Tools: Python 3.12 at **`/opt/venv/bin/python3`** (use this path; the plain `python3` on your PATH may lack the
packages) with numpy, scipy, OpenCV (`cv2`, headless), Pillow, matplotlib and MuJoCo 3.16. No network in your session.

## How you are graded (3 hidden shifts)

Gates (multiply the score): `contract_ok` = `busser.py` is a regular file that parses and defines `solve`, imports under
the run regime above (within 120 s and 120 CPU seconds), makes at least one `step` call over the shifts, holds no
remote network socket while it runs, your own `.py` files import no networking module (`socket`, `ssl`, `urllib`,
`http`, `requests`, `asyncio`, `ctypes`, `subprocess`, ...), load no module under a computed name, execute no computed
code (`exec` / `eval` / `compile` of a non-literal) and start no other program, the submission folder holds no model
weight file (a name ending in `.pt`, `.pth`, `.onnx`, `.safetensors`, `.ckpt`, `.h5` or `.pb`) and is at most 60 MB,
and no process of the program outlives its episode. Per shift `safety` (cell_manual.md section 10): a violation zeroes
that shift (touching a guest is one).

`service_score` = the mean over the 3 shifts (three fixed hidden shifts) of the checklist's credit less the damage
charges, / 10, read from the dining room's state when the shift ends (a program that stops early is scored then; the
tables whose party leaves later still count):
* tables turned, 6.0, shared equally by the shift's tables to turn (the ones left at the start and the ones whose party
  leaves before the shift's end); per table (its share x):
  * cleared 0.45: the share of its guests' things (each seat's plate, glass, fork, knife and napkin) that are off the
    table and in the bus tub, or taken by the dish crew;
  * wiped 0.25: half for the soil taken off, min(1, removed / 0.9), half for the top sanitized, min(1, covered / 0.9)
    (the share of the top the cloth passed over pressed while it held sanitizer, after the party left);
  * crumbs 0.05: the share of the table's crumbs caught in the crumb pan or the tub (or taken by the dish crew);
  * reset 0.25, once the table is cleared: the share of its seats with a roll-up lying in place: on the top in front of
    the seat, its middle 5 to 14 cm in from the edge and within 7 cm of the seat's middle, its axis within 20 degrees of
    square to the edge;
* the spill, 2.0: the cone standing upright beside it (0.3 to 1.2 m from the spill's middle, off the wet floor) within
  120 s of the spill 0.5 (within 300 s 0.25); the spill mopped up, 1.0 x min(1, removed / 0.95); the cone back on its
  spot at the end 0.25, if it stayed beside the spill until the floor was dry; the mop back in its clip 0.25, once it
  was used;
* the dish return, 1.0: the share of the turned tables' things that the dish crew took;
* closing, 1.0, for what was used during the shift: the cart at its home 0.4 (within 0.3 m, its heading within
  15 degrees; once it was pushed), the cloth in the sanitizer bucket 0.3 (once it wiped), the crumb pan on its hook
  0.3 (once it was taken off);
* damage, charged: 0.25 for each broken plate or glass (besides its lost credit), 0.25 for each thing left on the floor
  at the end (the guests' things, the roll-ups, the cloth, the crumb pan), 0.5 for each of a seated party's things a
  hand took; a shift's credit after the charges is at least 0.

The cell never tells your program what it has achieved; it reads the dining room's state when the shift ends. A
failure on the grading side (the cell, the renderer: a camera request answered `camera_unavailable`) is not scored as
your work: the run is then an infrastructure failure.

## Notes

- Images (camera frames) are meant to be processed by your program; look at the ones you need.
- The survey of the room, the furniture, the cart and the tools, the camera calibrations, the robot's kinematics and
  joint angles are exact; the localisation, the joint velocities and the joint torque estimates are slightly noisy;
  everything about the shift (the tables' states, the guests' things, the soil, the crumbs, the spill) must be measured
  in each episode.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is
  lost and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment.
  Write large files in several parts, keep every single response well below that limit, write a first complete
  submission early and refine it.
- Session time: this session is stopped after 4 hours of wall-clock time from its start. A session stopped by that limit
  is not graded at all, so a late finish scores nothing. Note the start time with `date` when you begin, keep a complete
  submission in place from early on, and end the session yourself (stop working and say that you are done) at least
  15 minutes before the limit.
