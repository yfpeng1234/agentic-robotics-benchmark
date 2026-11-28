"""The bistro's loose objects, the bus cart and the guests: geometry, mass, colliders, what the cameras see of them
(meshes.py, textures.py) and where a hand can take hold of them (grasp features).

Every loose-object builder adds its geoms to a body and returns the body's grasp features in the body's frame:
  ("cyl", a, b, r)            a cylinder or bar from point a to point b of radius r (glasses, cutlery handles,
                              roll-ups, the cart's push handle, the crumb pan's, the mop's and the cone's handles):
                              the hand's pinch closes across its axis;
  ("slab", c, half, k)        a box of half sizes `half` at c whose thin axis is k (the wiping cloth): the pinch
                              closes across its thickness, near an edge;
  ("disc", c, R, t)           a flat disc of radius R and half thickness t about the body's z axis (plates): the
                              pinch closes across the rim;
  ("ball", c, r)              a compact lump (crumpled napkins).
physics.Grips turns them into the cell's grip rule (the cell manual states it).

Collision bits (robot.py): 1 = the static room and its fixtures, 2 = the robot, 4 = loose objects and tools, 8 = the
floor, 16 = whatever a hand holds, 32 = the guests (they touch only the robot and what it holds: the cell's safety
watch counts any such touch).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import meshes as M
from . import props as P
from . import room as RM
from . import textures as T
from .robot import FLOOR, HELD, OBJ, ROBOT, WORLD

GUEST = 32
HELD_CART = 64                    # the cart while a hand holds it (it touches the room, the objects and the guests)
OBJ_CONTACT = dict(contype=OBJ, conaffinity=WORLD | ROBOT | OBJ | FLOOR, condim=3, group=3)
FIX_CONTACT = dict(contype=WORLD, conaffinity=ROBOT | OBJ | HELD, condim=3, group=3)
GUEST_CONTACT = dict(contype=GUEST, conaffinity=ROBOT | HELD | HELD_CART, condim=3, group=3)
VISUAL = dict(contype=0, conaffinity=0, group=1, mass=0.0)
G = mujoco.mjtGeom

# ------------------------------------------------------------------------------------------------ dimensions
PLATE = {"r": 0.120, "h": 0.022, "mass": 0.42}
GLASS = {"r": 0.035, "h": 0.115, "mass": 0.22}
FORK = {"len": 0.200, "mass": 0.045}
KNIFE = {"len": 0.215, "mass": 0.065}
NAPKIN = {"half": (0.045, 0.038, 0.022), "mass": 0.035}
ROLLUP = {"len": 0.23, "r": 0.021, "mass": 0.11}
CLOTH = {"size": (0.130, 0.090, 0.018), "mass": 0.070,
         # a folded bar towel gives under the hand: its pressed face (the pad) sinks into the fold against a spring
         # of 0.6 N per mm, by up to 10 mm either way (then it is pressed solid)
         "give": 0.010, "give_k": 600.0, "give_c": 8.0}
PAN = {"width": 0.22, "depth": 0.16, "wall": 0.05, "t": 0.003, "handle_len": 0.13, "handle_r": 0.011,
       "handle_tilt": math.radians(28.0), "mass": 0.22}
CONE = {"h": 0.82, "base": 0.30, "mass": 0.75, "bar_half": 0.055, "bar_r": 0.0125}
MOP = {"handle_len": 1.40, "handle_r": 0.0125, "head": (0.40, 0.11, 0.020), "pad": (0.42, 0.13, 0.014),
       "mass_handle": 0.35, "mass_head": 0.40, "swivel": math.radians(65.0)}

FOODS = ("pasta", "steak", "salad", "fish")
DRINKS = {"water": (0.80, 0.88, 0.92, 0.18), "iced_tea": (0.62, 0.36, 0.12, 0.62), "cola": (0.22, 0.10, 0.05, 0.85),
          "red_wine": (0.36, 0.04, 0.08, 0.85), "lemonade": (0.93, 0.88, 0.55, 0.55)}
SKIN = ((0.93, 0.76, 0.64), (0.80, 0.60, 0.46), (0.62, 0.43, 0.31), (0.45, 0.30, 0.22), (0.96, 0.84, 0.74))
HAIR = ((0.12, 0.09, 0.07), (0.32, 0.20, 0.12), (0.62, 0.48, 0.30), (0.75, 0.73, 0.70), (0.20, 0.14, 0.10))
SHIRT = ((0.20, 0.30, 0.45), (0.62, 0.30, 0.22), (0.38, 0.45, 0.33), (0.85, 0.83, 0.78), (0.55, 0.20, 0.30),
         (0.30, 0.32, 0.36), (0.80, 0.62, 0.30), (0.28, 0.48, 0.55))
TROUSERS = ((0.14, 0.15, 0.18), (0.22, 0.24, 0.30), (0.36, 0.30, 0.24), (0.10, 0.10, 0.11), (0.45, 0.42, 0.36))


def quat_from_R(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def quat_z(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


Q_X = quat_from_R(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float))   # a cylinder's axis (z) along x
Q_Y = quat_from_R(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float))   # ... along y


ROLE = mujoco.mjtTextureRole


def _pack_orm(rough, metal):
    out = np.zeros((4, 4, 3))
    out[..., 0], out[..., 1], out[..., 2] = 1.0, rough, metal
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8)


class Builder:
    """Adds meshes, materials and textures to a spec once each (names are unique). Materials are Filament's
    physically based ones: an albedo (a colour, or a texture of textures.py whose mesh coordinates are metres, `tile`
    metres per repeat) and an ORM map (ambient occlusion, roughness, metallic). Drawn-only geoms of static bodies are
    merged per material into one mesh (`static`, `flush`): far fewer renderables for the cameras."""

    def __init__(self, spec: mujoco.MjSpec):
        self.s = spec
        self.meshes = set()
        self.tex = {}
        self.mats = set()
        self.static_parts = {}

    def _texture(self, key, img, srgb):
        if key not in self.tex:
            img = np.ascontiguousarray(np.asarray(img)[..., :3])
            t = self.s.add_texture(name=f"tex_{len(self.tex)}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   width=int(img.shape[1]), height=int(img.shape[0]), nchannel=3,
                                   colorspace=(mujoco.mjtColorSpace.mjCOLORSPACE_SRGB if srgb
                                               else mujoco.mjtColorSpace.mjCOLORSPACE_LINEAR))
            t.data = np.ascontiguousarray(img[::-1]).tobytes()      # MuJoCo's row 0 is the image's bottom row
            self.tex[key] = t.name
        return self.tex[key]

    def material(self, name, rgba=(1, 1, 1, 1), tex=None, tile=None, rough=0.6, metal=0.0, emission=0.0):
        if name in self.mats:
            return name
        mt = self.s.add_material(name=name)
        for role in range(int(ROLE.mjNTEXROLE)):
            mt.textures[role] = ""
        mt.textures[ROLE.mjTEXROLE_ORM] = self._texture(("orm", round(rough, 3), round(metal, 3)),
                                                        _pack_orm(rough, metal), False)
        if tex:
            mt.textures[ROLE.mjTEXROLE_RGB] = self._texture(("img", tex), T.texture(tex), True)
            mt.texuniform = False
            rep = 1.0 / float(tile) if tile else 1.0
            mt.texrepeat = [rep, rep] if rgba[3] >= 1.0 else [1.0, 1.0]
        mt.rgba = [float(v) for v in rgba]
        mt.roughness, mt.metallic, mt.specular, mt.reflectance = 1.0, 1.0, 0.5, 0.0
        mt.emission = float(emission)
        self.mats.add(name)
        return name

    def mesh(self, name, mesh):
        if name not in self.meshes:
            V, F, N, UV = mesh
            m = self.s.add_mesh(name=name, inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            m.uservert = np.asarray(V, np.float64).ravel()
            m.usernormal = np.asarray(N, np.float64).ravel()
            m.usertexcoord = np.asarray(UV, np.float64).ravel()
            m.userface = np.asarray(F, np.int32).ravel()
            self.meshes.add(name)
        return name

    def visual(self, body, name, mesh, material, pos=(0, 0, 0), quat=(1, 0, 0, 0), mesh_name=None, group=1):
        kw = dict(VISUAL)
        kw["group"] = group
        mname = mesh if isinstance(mesh, str) else self.mesh(mesh_name or (name + "_mesh"), mesh)
        return body.add_geom(name=name, type=G.mjGEOM_MESH, meshname=mname, material=material, pos=list(pos),
                             quat=list(quat), **kw)

    def static(self, mesh, material, pos=(0.0, 0.0, 0.0), R=None):
        """A drawn-only part of the static room (world frame), merged per material at `flush`."""
        V, F, N, UV = mesh
        if R is not None:
            R = np.asarray(R, float)
            V, N = V @ R.T, N @ R.T
        self.static_parts.setdefault(material, []).append((V + np.asarray(pos, float), F, N, UV))

    def flush(self, body):
        for k, (mat, parts) in enumerate(sorted(self.static_parts.items())):
            self.visual(body, f"static_{k}", P.merge(*parts), mat, mesh_name=f"static_{k}_mesh")
        self.static_parts = {}

    def prim(self, body, name, gtype, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), *, material=None, rgba=None,
             visual=False, contact=OBJ_CONTACT, mass=None, friction=None, **kw):
        """A primitive geom: a collider (group 3, hidden: a visual mesh draws it) or, with visual=True, a drawn-only
        geom (group 1)."""
        args = dict(VISUAL) if visual else dict(contact)
        args.update(kw)
        if mass is not None and not visual:
            args["mass"] = mass
        if friction is not None:
            args["friction"] = list(friction)
        if material is not None:
            args["material"] = material
        if rgba is not None:
            args["rgba"] = list(rgba)
        if visual and kw.get("group") is None:
            args["group"] = 1
        return body.add_geom(name=name, type=gtype, size=list(size), pos=list(pos), quat=list(quat), **args)


# ================================================================================================ tableware
FOOD_MATS = {"sauce_red": (0.55, 0.14, 0.07, 1), "sauce_brown": (0.36, 0.20, 0.10, 1),
             "dressing": (0.82, 0.74, 0.42, 1), "sauce_cream": (0.93, 0.88, 0.72, 1), "pasta": (0.93, 0.80, 0.45, 1),
             "fries": (0.92, 0.72, 0.30, 1), "meat": (0.40, 0.22, 0.14, 1), "greens": (0.33, 0.55, 0.22, 1),
             "tomato": (0.80, 0.14, 0.08, 1), "lemon": (0.97, 0.86, 0.26, 1), "fish": (0.78, 0.64, 0.52, 1)}


def add_plate(b: Builder, body, name, p):
    R, h = PLATE["r"], PLATE["h"]
    b.prim(body, f"{name}_col", G.mjGEOM_CYLINDER, [R, h / 2, 0], (0, 0, h / 2), mass=PLATE["mass"],
           friction=(0.5, 0.005, 0.0001))
    pm, band = M.plate()
    b.visual(body, f"{name}_vis", pm, b.material("porcelain", (0.96, 0.96, 0.94, 1), rough=0.22), mesh_name="plate_mesh")
    b.visual(body, f"{name}_band", band, b.material("plate_band", (0.30, 0.42, 0.36, 1), rough=0.25), mesh_name="plate_band_mesh")
    if p.get("food"):
        for k, (mesh, mat) in enumerate(M.food(p["food"], int(p.get("seed", 0)))):
            b.visual(body, f"{name}_food{k}", mesh, b.material(f"food_{mat}", FOOD_MATS[mat]))
    return [("disc", (0, 0, h / 2), R, h / 2)]


def add_glass(b: Builder, body, name, p):
    r, h = GLASS["r"], GLASS["h"]
    b.prim(body, f"{name}_col", G.mjGEOM_CYLINDER, [0.033, h / 2, 0], (0, 0, h / 2), mass=GLASS["mass"],
           friction=(0.5, 0.005, 0.0001))
    b.visual(body, f"{name}_vis", M.tumbler(), b.material("glass", (0.88, 0.94, 0.97, 0.22), rough=0.05), mesh_name="tumbler_mesh")
    drink = p.get("drink")
    if drink:
        level = float(p.get("level", 0.04))
        col = DRINKS[drink]
        b.visual(body, f"{name}_drink", M.liquid(level), b.material(f"drink_{drink}", col, rough=0.1),
                 mesh_name=f"liquid_{int(level * 1000)}")
    return [("cyl", (0, 0, 0.012), (0, 0, h - 0.008), r)]


def add_fork(b: Builder, body, name, p):
    b.prim(body, f"{name}_col", G.mjGEOM_BOX, [FORK["len"] / 2, 0.010, 0.0035], (0, 0, 0.0035), mass=FORK["mass"],
           friction=(0.4, 0.005, 0.0001))
    b.visual(body, f"{name}_vis", M.fork(), b.material("steel", (0.80, 0.81, 0.83, 1), rough=0.28, metal=1.0), mesh_name="fork_mesh")
    return [("cyl", (-0.095, 0, 0.004), (-0.015, 0, 0.004), 0.006)]


def add_knife(b: Builder, body, name, p):
    b.prim(body, f"{name}_col", G.mjGEOM_BOX, [KNIFE["len"] / 2, 0.009, 0.0042], (-0.005, 0, 0.0042),
           mass=KNIFE["mass"], friction=(0.4, 0.005, 0.0001))
    b.visual(body, f"{name}_vis", M.knife(), b.material("steel", (0.80, 0.81, 0.83, 1), rough=0.28, metal=1.0), mesh_name="knife_mesh")
    return [("cyl", (-0.095, 0, 0.0045), (-0.010, 0, 0.0045), 0.007)]


def add_napkin(b: Builder, body, name, p):
    hx, hy, hz = NAPKIN["half"]
    b.prim(body, f"{name}_col", G.mjGEOM_ELLIPSOID, [hx, hy, hz], (0, 0, hz), mass=NAPKIN["mass"],
           friction=(0.8, 0.01, 0.0002))
    tex = "linen_sage" if p.get("colour", "sage") == "sage" else "linen_white"
    b.visual(body, f"{name}_vis", M.napkin(int(p.get("seed", 0))), b.material(f"napkin_{tex}", tex=tex, tile=0.1, rough=0.9))
    return [("ball", (0, 0, hz), 0.030)]


def add_rollup(b: Builder, body, name, p):
    L, r = ROLLUP["len"], ROLLUP["r"]
    # (a rolled napkin lies a little flattened: a box collider, which does not roll)
    b.prim(body, f"{name}_col", G.mjGEOM_BOX, [L / 2, r * 0.95, r * 0.88], (0, 0, r * 0.88), mass=ROLLUP["mass"],
           friction=(0.7, 0.01, 0.0002))
    nap, band = M.rollup(L, r)
    b.visual(body, f"{name}_vis", nap, b.material("rollup_linen", tex="linen_white", tile=0.1, rough=0.9), mesh_name="rollup_mesh")
    b.visual(body, f"{name}_band", band, b.material("rollup_band", tex="kraft", tile=0.1, rough=0.8), mesh_name="rollup_band_mesh")
    return [("cyl", (-L / 2 + 0.02, 0, r), (L / 2 - 0.02, 0, r), r)]


def add_cloth(b: Builder, body, name, p):
    lx, ly, lz = CLOTH["size"]
    m = CLOTH["mass"]
    # the cloth's body carries a sliver of its mass; its collider is the pad, sprung along the thickness (a folded
    # towel's give: pressed on a top it sinks into the fold, so a hand holding it presses with a few newtons over
    # several millimetres instead of a rigid block's hundreds)
    body.explicitinertial = True
    body.ipos = [0.0, 0.0, 0.0]
    body.iquat = [1.0, 0.0, 0.0, 0.0]
    body.mass = 0.15 * m
    body.inertia = [0.15 * m * (ly * ly + lz * lz) / 12, 0.15 * m * (lx * lx + lz * lz) / 12,
                    0.15 * m * (lx * lx + ly * ly) / 12]
    pad = body.add_body(name=f"{name}_pad")
    pad.add_joint(name=f"{name}_give", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                  range=[-CLOTH["give"], CLOTH["give"]], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                  stiffness=CLOTH["give_k"], damping=CLOTH["give_c"], springref=0.0, armature=0.0005,
                  solref_limit=[0.004, 1.0])
    # (a damp cloth slides easily on a varnished top)
    b.prim(pad, f"{name}_col", G.mjGEOM_BOX, [lx / 2, ly / 2, lz / 2], mass=0.85 * m,
           friction=(0.35, 0.005, 0.0001), priority=1)
    # drawn-only and without a material, so the cell can tint it as it soils (geom rgba)
    b.prim(body, f"{name}_vis", G.mjGEOM_BOX, [lx / 2 - 0.001, ly / 2 - 0.001, lz / 2 - 0.0005], visual=True,
           rgba=(0.86, 0.22, 0.20, 1))
    for k, y in enumerate((-ly / 4, ly / 4)):
        b.prim(body, f"{name}_stripe{k}", G.mjGEOM_BOX, [lx / 2, 0.006, lz / 2], (0, y, 0), visual=True,
               rgba=(0.95, 0.93, 0.88, 1))
    # a slab both ways: pinched through its thickness at an edge, or across its width from above (it then hangs flat
    # under the hand, its face down)
    return [("slab", (0, 0, 0), (lx / 2, ly / 2, lz / 2), 2), ("slab", (0, 0, 0), (lx / 2, ly / 2, lz / 2), 1)]


def pan_handle_axis():
    """(base point, unit axis) of the crumb pan's handle in the pan's frame."""
    D, H, tilt = PAN["depth"], PAN["wall"], PAN["handle_tilt"]
    return np.array([D - 0.005, 0.0, H * 0.7]), np.array([math.cos(tilt), 0.0, math.sin(tilt)])


def add_crumb_pan(b: Builder, body, name, p):
    """The crumb pan: frame at the middle of its lip, x from the lip to the back wall, z up, its floor on z = 0."""
    W, D, H, t = PAN["width"], PAN["depth"], PAN["wall"], PAN["t"]
    mp = PAN["mass"]
    b.prim(body, f"{name}_floor_col", G.mjGEOM_BOX, [D / 2, W / 2, t / 2], (D / 2, 0, t / 2), mass=mp * 0.45,
           friction=(0.5, 0.005, 0.0001))
    b.prim(body, f"{name}_back_col", G.mjGEOM_BOX, [t / 2, W / 2, H / 2], (D - t / 2, 0, H / 2), mass=mp * 0.2)
    for sgn in (-1, 1):
        b.prim(body, f"{name}_side{'pn'[sgn > 0]}_col", G.mjGEOM_BOX, [D * 0.36, t / 2, H * 0.34],
               (D * 0.62, sgn * (W / 2 - t / 2), H * 0.34), mass=mp * 0.1)
    pan, handle = M.crumb_pan(W, D, H, t)
    b.visual(body, f"{name}_pan", pan, b.material("pan_black", (0.07, 0.07, 0.08, 1)), mesh_name="crumb_pan_mesh")
    b.visual(body, f"{name}_handle", handle, b.material("pan_handle", (0.10, 0.10, 0.11, 1)),
             mesh_name="crumb_pan_handle_mesh")
    base, axis = pan_handle_axis()
    L = PAN["handle_len"]
    mid = base + axis * L / 2
    R = np.stack([np.cross([0, 1, 0], axis), [0, 1, 0], axis], axis=1)
    b.prim(body, f"{name}_handle_col", G.mjGEOM_CAPSULE, [PAN["handle_r"], L / 2, 0], tuple(mid), quat_from_R(R),
           mass=mp * 0.15)
    return [("cyl", tuple(base + axis * 0.03), tuple(base + axis * (L - 0.005)), PAN["handle_r"])]


def add_cone(b: Builder, body, name, p):
    """The wet-floor cone: frame at the middle of its base on the floor, z up; a T bar on top (along y) to carry it."""
    h = CONE["h"]
    m = CONE["mass"]
    b.prim(body, f"{name}_base_col", G.mjGEOM_BOX, [CONE["base"] / 2, CONE["base"] / 2, 0.015], (0, 0, 0.015),
           mass=m * 0.55, friction=(0.7, 0.01, 0.0002))
    for k, (r, z0, z1) in enumerate(((0.100, 0.03, 0.30), (0.072, 0.30, 0.55), (0.044, 0.55, h - 0.05))):
        b.prim(body, f"{name}_body{k}_col", G.mjGEOM_CYLINDER, [r, (z1 - z0) / 2, 0], (0, 0, (z0 + z1) / 2),
               mass=m * 0.12)
    b.prim(body, f"{name}_bar_col", G.mjGEOM_CAPSULE, [CONE["bar_r"], CONE["bar_half"], 0], (0, 0, h), Q_Y,
           mass=m * 0.09)
    shell, band, base, top = M.cone_body(h)
    yellow = b.material("cone_yellow", (0.95, 0.76, 0.10, 1), rough=0.45)
    b.visual(body, f"{name}_shell", shell, yellow, mesh_name="cone_shell_mesh")
    b.visual(body, f"{name}_band", band, b.material("cone_label", tex="cone_label", rough=0.45), mesh_name="cone_band_mesh")
    b.visual(body, f"{name}_base", base, b.material("cone_base", (0.10, 0.10, 0.10, 1)), mesh_name="cone_base_mesh")
    b.visual(body, f"{name}_top", top, b.material("cone_black", (0.08, 0.08, 0.08, 1)), mesh_name="cone_top_mesh")
    return [("cyl", (0, -CONE["bar_half"] + 0.01, h), (0, CONE["bar_half"] - 0.01, h), CONE["bar_r"])]


def add_mop(b: Builder, body, name, p):
    """The flat mop: `body` is the handle (its frame: origin on the swivel, z along the handle); the head hangs on a
    two-axis swivel (child body), so its pad stays flat on the floor as the handle leans."""
    L, r = MOP["handle_len"], MOP["handle_r"]
    b.prim(body, f"{name}_handle_col", G.mjGEOM_CAPSULE, [r, (L - 0.06) / 2, 0], (0, 0, (L + 0.06) / 2),
           mass=MOP["mass_handle"], friction=(0.6, 0.005, 0.0001))
    (hm, grip), (frame, pad) = M.mop_parts(L)
    b.visual(body, f"{name}_handle", hm, b.material("mop_alu", (0.72, 0.74, 0.77, 1), rough=0.35, metal=1.0), mesh_name="mop_handle_mesh")
    b.visual(body, f"{name}_grip", grip, b.material("mop_grip", (0.12, 0.30, 0.62, 1)), mesh_name="mop_grip_mesh")
    gimbal = body.add_body(name=f"{name}_gimbal", pos=[0, 0, 0])
    gimbal.add_joint(name=f"{name}_swivel_y", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0],
                     range=[-MOP["swivel"], MOP["swivel"]], limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.4,
                     armature=0.0005)
    gimbal.add_geom(name=f"{name}_gimbal_mass", type=G.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0,
                    group=4, mass=0.02)
    head = gimbal.add_body(name=f"{name}_head", pos=[0, 0, 0])
    head.add_joint(name=f"{name}_swivel_x", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0],
                   range=[-MOP["swivel"], MOP["swivel"]], limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.4,
                   armature=0.0005)
    hx, hy, hz = MOP["head"]
    px, py, pz = MOP["pad"]
    b.prim(head, f"{name}_frame_col", G.mjGEOM_BOX, [hx / 2, hy / 2, hz / 2], (0, 0, -0.022),
           mass=MOP["mass_head"] * 0.6, friction=(0.5, 0.005, 0.0001))
    # the pad on its sprung plate: it gives against a spring of 0.5 N per mm, by up to 20 mm either way (the head
    # follows the floor under a hand that holds the handle a little low or high); the microfibre slippery on the floor
    padb = head.add_body(name=f"{name}_padbody", pos=[0, 0, 0])
    padb.add_joint(name=f"{name}_pad_give", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], range=[-0.020, 0.020],
                   limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=500.0, damping=8.0, springref=0.0,
                   armature=0.0005, solref_limit=[0.004, 1.0])
    b.prim(padb, f"{name}_pad_col", G.mjGEOM_BOX, [px / 2, py / 2, pz / 2], (0, 0, -0.039),
           mass=MOP["mass_head"] * 0.4, friction=(0.15, 0.005, 0.0001), priority=1)
    b.visual(head, f"{name}_frame", frame, b.material("mop_frame", (0.15, 0.17, 0.20, 1)), mesh_name="mop_frame_mesh")
    b.visual(head, f"{name}_pad", pad, b.material("mop_pad", tex="mop_pad", tile=0.1, rough=0.95), mesh_name="mop_pad_mesh")
    return [("cyl", (0, 0, 0.35), (0, 0, L - 0.01), r)]


BUILDERS = {"plate": add_plate, "glass": add_glass, "fork": add_fork, "knife": add_knife, "napkin": add_napkin,
            "rollup": add_rollup, "cloth": add_cloth, "crumb_pan": add_crumb_pan, "cone": add_cone, "mop": add_mop}
# what a guest leaves at their cover (the busser clears it), the reset item, the tools
GUEST_KINDS = ("plate", "glass", "fork", "knife", "napkin")
FRAGILE_KINDS = ("plate", "glass")


# ================================================================================================ the bus cart
def add_cart(b: Builder, wb, pose):
    """The bus cart: a planar body (slides along x and y, turns about z) at `pose` (x, y, yaw); its frame on the floor
    under the top shelf's middle, x towards the front. Its top shelf carries the bus tub (front) and the sanitizer
    bucket (back); the crumb pan hangs below the push handle. Returns its record (grasp features of the push handle,
    the tub's inside box, the bucket in the cart's frame)."""
    C = RM.CART
    body = wb.add_body(name="cart", pos=[0, 0, 0])
    body.add_joint(name="cart_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], damping=4000.0, armature=1.0)
    body.add_joint(name="cart_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], damping=4000.0, armature=1.0)
    body.add_joint(name="cart_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=1500.0, armature=0.5)
    sx, sy = C["size"]
    tz, lz, lip = C["top_z"], C["lower_z"], C["lip"]
    ct = dict(FIX_CONTACT)
    ct["contype"] = WORLD
    m_total = C["mass"]
    # colliders: the two shelves with their lips, the posts, the casters' blocks, the handle
    for k, z in enumerate((tz, lz)):
        b.prim(body, f"cart_shelf{k}_col", G.mjGEOM_BOX, [sx / 2, sy / 2, 0.016], (0, 0, z - 0.016), contact=ct,
               mass=m_total * 0.3, friction=(0.6, 0.005, 0.0001))
        for sgn in (-1, 1):
            b.prim(body, f"cart_lip{k}{'pn'[sgn > 0]}y_col", G.mjGEOM_BOX, [sx / 2, 0.006, lip / 2],
                   (0, sgn * (sy / 2 - 0.006), z + lip / 2), contact=ct, mass=0.2)
            b.prim(body, f"cart_lip{k}{'pn'[sgn > 0]}x_col", G.mjGEOM_BOX, [0.006, sy / 2, lip / 2],
                   (sgn * (sx / 2 - 0.006), 0, z + lip / 2), contact=ct, mass=0.2)
    px, py = sx / 2 - 0.035, sy / 2 - 0.035
    for i, x in enumerate((-px, px)):
        for j, y in enumerate((-py, py)):
            b.prim(body, f"cart_post{i}{j}_col", G.mjGEOM_CYLINDER, [0.016, (tz - 0.10) / 2, 0],
                   (x, y, 0.10 + (tz - 0.10) / 2), contact=ct, mass=0.6)
            b.prim(body, f"cart_caster{i}{j}_col", G.mjGEOM_BOX, [0.035, 0.025, 0.05], (x, y, 0.05), contact=ct,
                   mass=0.3)
    hz, hx, hh, hr = C["handle_z"], C["handle_x"], C["handle_half"], C["handle_r"]
    b.prim(body, "cart_handle_col", G.mjGEOM_CAPSULE, [hr, hh, 0], (hx, 0, hz), Q_Y, contact=ct, mass=0.4)
    for sgn in (-1, 1):
        b.prim(body, f"cart_handle_leg{'pn'[sgn > 0]}_col", G.mjGEOM_CAPSULE, [hr, 0.06, 0],
               ((hx - sx / 2) / 2, sgn * hh, (hz + tz) / 2), quat_from_R(_dir_R((-sx / 2 - hx, 0, tz - hz))),
               contact=ct, mass=0.1)
    shelves, frame, wheels, (handle, grip) = M.cart(C)
    b.visual(body, "cart_shelves", shelves, b.material("cart_poly", (0.09, 0.09, 0.10, 1)))
    b.visual(body, "cart_frame", frame, b.material("cart_steel", (0.62, 0.64, 0.66, 1), rough=0.35, metal=1.0))
    b.visual(body, "cart_wheels", wheels, b.material("cart_wheel", (0.05, 0.05, 0.05, 1)))
    b.visual(body, "cart_handle", handle, b.material("cart_steel", (0.62, 0.64, 0.66, 1), rough=0.35, metal=1.0))
    b.visual(body, "cart_grip", grip, b.material("cart_grip", (0.06, 0.06, 0.07, 1)))
    # the bus tub (part of the cart: the dish crew empties it in place)
    TB = RM.TUB
    (ox, oy), h, wt = TB["outer"], TB["h"], TB["wall_t"]
    tx, ty = C["tub_at"]
    z0 = tz
    b.prim(body, "tub_floor_col", G.mjGEOM_BOX, [ox / 2, oy / 2, wt / 2], (tx, ty, z0 + wt / 2), contact=ct,
           mass=TB["mass"] * 0.4, friction=(0.6, 0.005, 0.0001))
    for sgn in (-1, 1):
        b.prim(body, f"tub_wall{'pn'[sgn > 0]}y_col", G.mjGEOM_BOX, [ox / 2, wt / 2, h / 2],
               (tx, ty + sgn * (oy / 2 - wt / 2), z0 + h / 2), contact=ct, mass=TB["mass"] * 0.15)
        b.prim(body, f"tub_wall{'pn'[sgn > 0]}x_col", G.mjGEOM_BOX, [wt / 2, oy / 2 - wt, h / 2],
               (tx + sgn * (ox / 2 - wt / 2), ty, z0 + h / 2), contact=ct, mass=TB["mass"] * 0.15)
    b.visual(body, "tub", M.tub(ox, oy, h, wt), b.material("tub_grey", (0.42, 0.44, 0.45, 1), rough=0.55), pos=(tx, ty, z0))
    # the sanitizer bucket (its solution drawn as a surface; the cloth lives in it)
    BK = RM.BUCKET
    bx, by = C["bucket_at"]
    nseg = 10
    for k in range(nseg):
        a = 2 * math.pi * k / nseg
        b.prim(body, f"bucket_wall{k}_col", G.mjGEOM_BOX, [0.006, BK["r"] * math.tan(math.pi / nseg), BK["h"] / 2],
               (bx + (BK["r"] - 0.006) * math.cos(a), by + (BK["r"] - 0.006) * math.sin(a), z0 + BK["h"] / 2),
               quat_z(a), contact=ct, mass=0.03)
    b.prim(body, "bucket_floor_col", G.mjGEOM_CYLINDER, [BK["r"] - 0.008, 0.004, 0], (bx, by, z0 + 0.004),
           contact=ct, mass=0.1)
    # the rack's collider: two crossed square plates inscribed in the pail (a flat face under the cloth: a box resting
    # on a cylinder touches it at a single point)
    hs = (BK["r"] - 0.010) / math.sqrt(2.0)
    for k, a in enumerate((0.0, math.pi / 4)):
        b.prim(body, "bucket_rack_col" if k == 0 else "bucket_rack2_col", G.mjGEOM_BOX, [hs, hs, 0.003],
               (bx, by, z0 + BK["rack_h"] - 0.003), quat_z(a), contact=ct, mass=0.025)
    b.prim(body, "bucket_rack", G.mjGEOM_CYLINDER, [BK["r"] - 0.011, 0.002, 0], (bx, by, z0 + BK["rack_h"] - 0.003),
           visual=True, material=b.material("rack_grey", (0.55, 0.57, 0.58, 1), rough=0.5, metal=0.5))
    pail, bail = M.bucket(BK["r"], BK["h"], BK["wall_t"])
    b.visual(body, "bucket", pail, b.material("bucket_red", (0.72, 0.09, 0.08, 1), rough=0.45), pos=(bx, by, z0))
    b.visual(body, "bucket_bail", bail, b.material("cart_steel", (0.62, 0.64, 0.66, 1), rough=0.35, metal=1.0), pos=(bx, by, z0))
    b.prim(body, "sanitizer", G.mjGEOM_CYLINDER, [BK["r"] - 0.012, 0.001, 0], (bx, by, z0 + BK["fill_h"]),
           visual=True, rgba=(0.55, 0.80, 0.78, 0.55))
    # the crumb pan's hook (drawn)
    hxp, hyp, hzp = C["pan_hook"]
    b.prim(body, "pan_hook", G.mjGEOM_CAPSULE, [0.005, 0.02, 0], (hxp - 0.01, hyp, hzp + 0.01), Q_X, visual=True,
           material=b.material("cart_steel", (0.62, 0.64, 0.66, 1), rough=0.35, metal=1.0))
    feats = [("cyl", (hx, -hh + 0.03, hz), (hx, hh - 0.03, hz), hr)]
    tub_inside = (tx - ox / 2 + wt, ty - oy / 2 + wt, z0 + wt, tx + ox / 2 - wt, ty + oy / 2 - wt, z0 + h)
    return {"body": body, "features": feats, "tub_inside": tub_inside,
            "bucket": (bx, by, z0, BK["r"], BK["fill_h"]), "pose": pose}


def _dir_R(v):
    """A rotation whose z axis is along v."""
    v = np.asarray(v, float)
    z = v / np.linalg.norm(v)
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


# ================================================================================================ the guests
def add_guest(b: Builder, wb, seat, look, name):
    """A seated guest at `seat` (room.tables() seat record) as a mocap body: an architectural-visualisation figure
    (smooth, matte; skin, hair, clothes) built from primitives that are also its colliders (bit GUEST: they touch
    only the robot and what it holds). Frame: on the floor under the chair's middle, x the way the guest faces."""
    x, y = seat["chair"]
    body = wb.add_body(name=name, pos=[x, y, 0.0], quat=list(quat_z(seat["facing"])), mocap=True)
    skin = b.material(f"skin{look['skin']}", SKIN[look["skin"]] + (1.0,), rough=0.7)
    hair = b.material(f"hair{look['hair']}", HAIR[look["hair"]] + (1.0,), rough=0.8)
    shirt = b.material(f"shirt{look['shirt']}", SHIRT[look["shirt"]] + (1.0,), rough=0.85)
    pants = b.material(f"pants{look['pants']}", TROUSERS[look["pants"]] + (1.0,), rough=0.85)
    shoes = b.material("shoes", (0.08, 0.07, 0.07, 1.0))
    s = look.get("scale", 1.0)
    w = look.get("build", 1.0)

    def geom(nm, gtype, size, pos, mat, quat=(1, 0, 0, 0)):
        body.add_geom(name=f"{name}_{nm}", type=gtype, size=[v * s for v in size],
                      pos=[pos[0] * s, pos[1] * s, pos[2]], quat=list(quat), material=mat, group=1, mass=0.0,
                      **{k: v for k, v in GUEST_CONTACT.items() if k != "group"})

    lean = look.get("lean", 0.0)
    geom("pelvis", G.mjGEOM_ELLIPSOID, [0.13, 0.17 * w, 0.09], (-0.04, 0, 0.55), pants)
    geom("torso", G.mjGEOM_ELLIPSOID, [0.11, 0.18 * w, 0.24], (-0.03 + lean, 0, 0.82), shirt)
    geom("neck", G.mjGEOM_CAPSULE, [0.045, 0.04, 0], (0.0 + lean, 0, 1.07), skin)
    geom("head", G.mjGEOM_ELLIPSOID, [0.090, 0.078, 0.108], (0.01 + lean, 0, 1.19), skin)
    geom("hair", G.mjGEOM_ELLIPSOID, [0.094, 0.082, 0.075], (-0.006 + lean, 0, 1.235), hair)
    for sgn, sd in ((1, "l"), (-1, "r")):
        sh = np.array([-0.03 + lean, sgn * 0.19 * w, 1.00])
        el = np.array([0.08, sgn * 0.23 * w, 0.78])
        hd = np.array([0.27, sgn * 0.21, 0.775])
        geom(f"upper_{sd}", G.mjGEOM_CAPSULE, [0.048, np.linalg.norm(el - sh) / 2, 0], tuple((sh + el) / 2), shirt,
             quat_from_R(_dir_R(el - sh)))
        geom(f"fore_{sd}", G.mjGEOM_CAPSULE, [0.040, np.linalg.norm(hd - el) / 2, 0], tuple((el + hd) / 2), shirt,
             quat_from_R(_dir_R(hd - el)))
        geom(f"hand_{sd}", G.mjGEOM_ELLIPSOID, [0.050, 0.035, 0.022], tuple(hd + np.array([0.04, 0, 0])), skin)
        hip = np.array([-0.02, sgn * 0.095 * w, 0.55])
        knee = np.array([0.33, sgn * 0.10 * w, 0.56])
        ank = np.array([0.38, sgn * 0.11 * w, 0.09])
        geom(f"thigh_{sd}", G.mjGEOM_CAPSULE, [0.072, np.linalg.norm(knee - hip) / 2, 0], tuple((hip + knee) / 2),
             pants, quat_from_R(_dir_R(knee - hip)))
        geom(f"shin_{sd}", G.mjGEOM_CAPSULE, [0.050, np.linalg.norm(ank - knee) / 2, 0], tuple((knee + ank) / 2),
             pants, quat_from_R(_dir_R(ank - knee)))
        geom(f"foot_{sd}", G.mjGEOM_ELLIPSOID, [0.12, 0.048, 0.038], (0.44, sgn * 0.11 * w, 0.038), shoes)
    return body
