# Survey photos

Camera frames of two development shifts of the bistro (`shift_a`, `shift_b`: cell seeds 101 and 102, which are not in
the graded hidden set), rendered by the grader's own cell (`tests/bistro_cell`, the same renderer and sensor model as a
graded episode) with `get_camera`, the robot standing idle (the head frames from the robot set down at the spots
`index.json` names). Per frame: `<name>.json` (the reply's metadata), `<name>_rgb.png` (colour) and
`<name>_depth_mm.png` (16-bit depth in mm, 0 = no reading). The image build copies this folder to
`/opt/arb-assets/survey`; `environment/setup/setup.sh` stages it into the agent home (`~/given/cell/survey`) and
`tests/test.sh` into the grader's root-owned copy of the givens (`given_hashes.json` lists every file).
