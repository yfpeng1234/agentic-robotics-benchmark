"""MuJoCo model of the bistro cell: the dining room (shell, street front, kitchen wall, service station, dish-return
pass), the tables and chairs, the guests (mocap figures), the bus cart, the loose objects (free bodies), the crumbs
(drawn-only mocap bodies), the soiling of the tables and the floor spill (drawn-only cells the cell colours), one G1
humanoid, the ceiling cameras and the inactive weld pool of the grips and the clips.

Everything static that the cameras see is merged per material into a few meshes (objects.Builder.static); the
colliders are primitive shapes (contype 1). Units: metres, radians; room frame of room.py.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import meshes as M
from . import objects as O
from . import props as P
from . import room as RM
from .render import filament_view_keys
from .robot import FLOOR, HELD, OBJ, ROBOT, WORLD, g1_spec

G = mujoco.mjtGeom
ROBOT_NAME = "a"
STATIC = dict(contype=WORLD, conaffinity=ROBOT | OBJ | HELD, condim=3, group=3, friction=[0.6, 0.005, 0.0001])
SUN_DIR = (0.30, 0.62, -0.72)          # the sun at lunchtime, from the south-south-west, about 46 deg high
CRUMB_COLOURS = ((0.80, 0.62, 0.40), (0.70, 0.50, 0.30), (0.90, 0.78, 0.50), (0.55, 0.38, 0.22), (0.93, 0.88, 0.76))
SOIL_RGB = (0.38, 0.20, 0.09)
SPILL_RGB = {"cola": (0.20, 0.09, 0.04), "orange_juice": (0.90, 0.55, 0.12), "red_wine": (0.32, 0.03, 0.07)}
PARK_AT = (3.0, 16.0, 0.5)             # where the cell keeps the bodies not in play (far north, out of every view)


def quat_z(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def _look_at_quat(pos, target):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    zc = pos - target
    zc /= np.linalg.norm(zc)
    xc = np.cross([0.0, 0.0, 1.0], zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return _quat(np.stack([xc, yc, zc], axis=1))


def collider_box(body, name, box, contact=STATIC, **kw):
    x0, y0, z0, x1, y1, z1 = box
    args = dict(contact)
    args.update(kw)
    return body.add_geom(name=name, type=G.mjGEOM_BOX, size=[(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2],
                         pos=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], **args)


def slab(B, box, mat, radius=0.0, n=2):
    """A drawn box (x0, y0, z0, x1, y1, z1) in the room, merged into the static meshes."""
    x0, y0, z0, x1, y1, z1 = box
    m = P.rounded_box([(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2], radius, n if radius > 0 else 1)
    B.static(m, mat, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


# ================================================================================================ the shell
def add_shell(B: O.Builder):
    s = B.s
    wb = s.worldbody
    X, Y, H, t = RM.ROOM_X, RM.ROOM_Y, RM.WALL_H, RM.WALL_T
    wb.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[40, 40, 0.1], pos=[0, 0, 0], contype=FLOOR,
                conaffinity=OBJ | HELD, group=3, friction=[0.6, 0.005, 0.0001])
    floor = B.material("floor", tex="floor", tile=1.2, rough=0.55)
    slab(B, (0.0, 0.0, -0.02, X, Y, 0.0), floor)
    plaster = B.material("plaster", tex="plaster", tile=1.5, rough=0.92)
    wains = B.material("wainscot", tex="wainscot", tile=1.0, rough=0.55)
    dark = B.material("trim_dark", (0.10, 0.14, 0.12, 1), rough=0.5)
    cream = B.material("trim_cream", (0.88, 0.84, 0.76, 1), rough=0.6)
    WZ = 1.05                                 # the wainscot's height
    # colliders: the four walls (openings where the windows, the doors and the pass are)
    sw = [(0.0, 0.40)] + [(a, b) for a, b in zip([w[1] for w in RM.WINDOWS], [w[0] for w in RM.WINDOWS[1:]])] + \
        [(RM.WINDOWS[-1][1], RM.ENTRANCE_X[0]), (RM.ENTRANCE_X[1], X)]
    for k, (a, b) in enumerate(sw):
        collider_box(wb, f"wall_s{k}", (a, -t, 0.0, b, 0.0, H))
    for k, (a, b) in enumerate(RM.WINDOWS):
        collider_box(wb, f"wall_s_sill{k}", (a, -t, 0.0, b, 0.0, RM.WINDOW_Z[0]))
        collider_box(wb, f"wall_s_head{k}", (a, -t, RM.WINDOW_Z[1], b, 0.0, H))
    collider_box(wb, "wall_s_door_head", (RM.ENTRANCE_X[0], -t, RM.ENTRANCE_H, RM.ENTRANCE_X[1], 0.0, H))
    collider_box(wb, "wall_s_door", (RM.ENTRANCE_X[0], -t - 0.05, 0.0, RM.ENTRANCE_X[1], -t, RM.ENTRANCE_H))
    kd, ps = RM.KITCHEN_DOOR_X, RM.PASS["counter"]
    nw = [(0.0, kd[0]), (kd[1], ps[0]), (ps[3], X)]
    for k, (a, b) in enumerate(nw):
        collider_box(wb, f"wall_n{k}", (a, Y, 0.0, b, Y + t, H))
    collider_box(wb, "wall_n_door", (kd[0], Y, 0.0, kd[1], Y + t, H))            # the kitchen door (closed)
    collider_box(wb, "wall_n_pass_head", (ps[0], Y, RM.PASS["opening_z"][1], ps[3], Y + t, H))
    collider_box(wb, "wall_w", (-t, 0.0, 0.0, 0.0, Y, H))
    collider_box(wb, "wall_e", (X, 0.0, 0.0, X + t, Y, H))
    # drawn: wainscot below, plaster above, the chair rail, the skirting, the crown
    def wall_faces(axis, at, lo, hi, holes, inward):
        """Drawn wall faces along `axis` ('x' for the south and north walls) from lo to hi, with holes [(a, b, z0,
        z1)]: per strip of the wall between hole edges, the wainscot and the plaster where nothing is cut out."""
        edges = sorted({lo, hi} | {h[0] for h in holes} | {h[1] for h in holes})
        for a, b in zip(edges[:-1], edges[1:]):
            cut = [h for h in holes if h[0] <= a + 1e-9 and h[1] >= b - 1e-9]
            zs = [(0.0, H)]
            for h in cut:
                zs = [seg for z0, z1 in zs for seg in ((z0, min(z1, h[2])), (max(z0, h[3]), z1)) if seg[1] - seg[0] > 1e-6]
            for z0, z1 in zs:
                for (za, zb, mat, d) in ((z0, min(z1, WZ), wains, 0.012), (max(z0, WZ), z1, plaster, 0.004)):
                    if zb - za < 1e-6:
                        continue
                    if axis == "x":
                        y0, y1 = (at, at + inward * d) if inward > 0 else (at + inward * d, at)
                        slab(B, (a, y0, za, b, y1, zb), mat)
                    else:
                        x0, x1 = (at, at + inward * d) if inward > 0 else (at + inward * d, at)
                        slab(B, (x0, a, za, x1, b, zb), mat)
    south_holes = [(a, b, RM.WINDOW_Z[0], RM.WINDOW_Z[1]) for a, b in RM.WINDOWS] + \
        [(RM.ENTRANCE_X[0], RM.ENTRANCE_X[1], 0.0, RM.ENTRANCE_H)]
    wall_faces("x", 0.0, 0.0, X, south_holes, +1)
    north_holes = [(kd[0], kd[1], 0.0, RM.KITCHEN_DOOR_H), (ps[0], ps[3], RM.PASS["opening_z"][0],
                                                            RM.PASS["opening_z"][1])]
    wall_faces("x", Y, 0.0, X, north_holes, -1)
    wall_faces("y", 0.0, 0.0, Y, [], +1)
    wall_faces("y", X, 0.0, Y, [], -1)
    # trims
    for (a, b, holes, y, sgn) in ((0.0, X, south_holes, 0.0, 1), (0.0, X, north_holes, Y, -1)):
        edges = sorted({a, b} | {h[0] for h in holes} | {h[1] for h in holes})
        for u, v in zip(edges[:-1], edges[1:]):
            cut = [h for h in holes if h[0] <= u + 1e-9 and h[1] >= v - 1e-9]
            for (z0, z1, d, mat) in ((0.0, 0.12, 0.022, dark), (WZ - 0.01, WZ + 0.035, 0.03, dark),
                                     (H - 0.09, H, 0.05, cream)):
                if any(h[2] < z1 and h[3] > z0 for h in cut):
                    continue
                yy = (y, y + sgn * d) if sgn > 0 else (y + sgn * d, y)
                slab(B, (u, yy[0], z0, v, yy[1], z1), mat)
    for (x, sgn) in ((0.0, 1), (X, -1)):
        for (z0, z1, d, mat) in ((0.0, 0.12, 0.022, dark), (WZ - 0.01, WZ + 0.035, 0.03, dark),
                                 (H - 0.09, H, 0.05, cream)):
            xx = (x, x + sgn * d) if sgn > 0 else (x + sgn * d, x)
            slab(B, (xx[0], 0.0, z0, xx[1], Y, z1), mat)
    # ceiling and its beams
    ceil = B.material("ceiling", tex="ceiling", tile=2.0, rough=0.95)
    slab(B, (-t, -t, H, X + t, Y + t, H + 0.05), ceil)
    beam = B.material("beam", tex="beam", tile=1.0, rough=0.7)
    for xb in (1.30, 3.90, 6.50, 9.10):
        slab(B, (xb - 0.09, 0.0, H - 0.24, xb + 0.09, Y, H), beam, 0.01)
    add_windows(B)
    add_doors(B)


def add_windows(B: O.Builder):
    steel = B.material("window_steel", (0.07, 0.075, 0.08, 1), rough=0.45, metal=0.6)
    glass = B.material("window_glass", (0.80, 0.88, 0.92, 0.10), rough=0.02)
    sill = B.material("sill_oak", tex="oak", tile=0.8, rough=0.5)
    z0, z1 = RM.WINDOW_Z
    t = RM.WALL_T
    rod = B.material("brass", (0.80, 0.62, 0.32, 1), rough=0.3, metal=1.0)
    curtain = B.material("curtain", (1.0, 1.0, 1.0, 0.86), tex="linen_curtain", rough=0.95)
    for a, b in RM.WINDOWS:
        fw = 0.045
        for box in ((a, -t + 0.03, z0, a + fw, -0.03, z1), (b - fw, -t + 0.03, z0, b, -0.03, z1),
                    (a, -t + 0.03, z1 - fw, b, -0.03, z1), (a, -t + 0.03, z0, b, -0.03, z0 + fw)):
            slab(B, box, steel)
        n = 3
        for k in range(1, n):
            xm = a + (b - a) * k / n
            slab(B, (xm - 0.015, -t + 0.05, z0, xm + 0.015, -0.05, z1), steel)
        for zm in (1.40, 2.25):
            slab(B, (a, -t + 0.05, zm - 0.015, b, -0.05, zm + 0.015), steel)
        slab(B, (a - 0.03, 0.0, z0 - 0.03, b + 0.03, 0.14, z0 + 0.012), sill, 0.006)
        # cafe curtains on a brass rod (drawn): the lower panes
        slab(B, (a + 0.02, 0.020, 1.37, b - 0.02, 0.030, 1.385), rod, 0.004)
        L = b - a - 0.06
        cm = P.transform(P.quad(L, 1.36 - z0 - 0.04, "+y", 0.0, L / 0.25, 0.0, (1.36 - z0 - 0.04) / 0.25),
                         t=((a + b) / 2, 0.028, (z0 + 0.04 + 1.36) / 2))
        B.static(cm, curtain)
        B.static(P.transform(P.quad(L, 1.36 - z0 - 0.04, "-y", 0.0, L / 0.25, 0.0, (1.36 - z0 - 0.04) / 0.25),
                             t=((a + b) / 2, 0.027, (z0 + 0.04 + 1.36) / 2)), curtain)
    # outside: the pavement and the street across (a soft backdrop lit by the day)
    pave = B.material("pavement", (0.56, 0.55, 0.53, 1), rough=0.9)
    slab(B, (-6.0, -7.0, -0.04, RM.ROOM_X + 6.0, -t, -0.02), pave)
    street = B.material("street", (0.80, 0.80, 0.80, 1.0), tex="street", rough=1.0, emission=0.12)
    sq = P.quad(34.0, 12.0, "+y", 0.0, 1.0, 0.0, 1.0)
    B.static(P.transform(sq, t=(RM.ROOM_X / 2, -9.0, 4.0)), street)


def add_doors(B: O.Builder):
    t = RM.WALL_T
    steel = B.material("window_steel", (0.07, 0.075, 0.08, 1), rough=0.45, metal=0.6)
    glass = B.material("window_glass", (0.80, 0.88, 0.92, 0.10), rough=0.02)
    brass = B.material("brass", (0.80, 0.62, 0.32, 1), rough=0.3, metal=1.0)
    a, b = RM.ENTRANCE_X
    h = RM.ENTRANCE_H
    for box in ((a, -t, 0.0, a + 0.06, -0.02, h), (b - 0.06, -t, 0.0, b, -0.02, h), (a, -t, h - 0.08, b, -0.02, h),
                (a, -t, 0.0, b, -0.02, 0.18), (a + (b - a) / 2 - 0.02, -t, 0.0, a + (b - a) / 2 + 0.02, -0.02, h)):
        slab(B, box, steel)
    for xh in (a + (b - a) / 2 - 0.08, a + (b - a) / 2 + 0.08):
        slab(B, (xh - 0.012, -0.02, 0.85, xh + 0.012, 0.03, 1.25), brass, 0.008)
    # the kitchen door: a stained oak leaf with a porthole and a kick plate
    k0, k1 = RM.KITCHEN_DOOR_X
    door = B.material("door_oak", tex="walnut", tile=1.0, rough=0.5)
    Y = RM.ROOM_Y
    slab(B, (k0 - 0.06, Y - 0.03, 0.0, k1 + 0.06, Y, RM.KITCHEN_DOOR_H + 0.06), B.material("trim_dark", (0.10, 0.14,
                                                                                               0.12, 1), rough=0.5))
    slab(B, (k0, Y - 0.04, 0.0, k1, Y - 0.005, RM.KITCHEN_DOOR_H), door, 0.004)
    port = P.transform(P.lathe([(0.0, 0.0), (0.13, 0.0), (0.13, 0.012), (0.0, 0.012)], 32), R=M.Rx(math.pi / 2),
                       t=((k0 + k1) / 2, Y - 0.04, 1.55))
    B.static(port, steel)
    B.static(P.transform(P.lathe([(0.0, 0.0), (0.105, 0.0), (0.105, 0.004), (0.0, 0.004)], 32), R=M.Rx(math.pi / 2),
                         t=((k0 + k1) / 2, Y - 0.046, 1.55)), B.material("porthole", (0.30, 0.34, 0.36, 1),
                                                                          rough=0.1))
    slab(B, (k0 + 0.02, Y - 0.045, 0.02, k1 - 0.02, Y - 0.04, 0.28), B.material("stainless", tex="stainless",
                                                                                 tile=0.5, rough=0.35, metal=1.0))


# ================================================================================================ the dining room
def add_tables(B: O.Builder, layout):
    wb = B.s.worldbody
    top_m = B.material("walnut", tex="walnut", tile=0.9, rough=0.42)
    iron = B.material("cast_iron", (0.06, 0.06, 0.065, 1), rough=0.55, metal=0.5)
    bent = B.material("bentwood", tex="bentwood", tile=0.3, rough=0.5)
    cane = B.material("cane", tex="cane", tile=1.0, rough=0.7)
    brass = B.material("brass", (0.80, 0.62, 0.32, 1), rough=0.3, metal=1.0)
    shade_in = B.material("shade_inside", (0.95, 0.90, 0.80, 1), rough=0.5, emission=0.25)
    bulb = B.material("bulb", (1.0, 0.92, 0.78, 1), rough=0.3, emission=3.0)
    frame, seat = M.bentwood_chair(RM.CHAIR["seat_z"])
    pend, pend_in, pend_bulb = M.pendant(1.10, 0.13)
    zt, tt = RM.TABLE_TOP_Z, RM.TABLE_TOP_T
    for name, t in RM.tables().items():
        (cx, cy), (sx, sy) = t["center"], t["size"]
        body = wb.add_body(name=f"table_{name}", pos=[cx, cy, 0.0])
        collider_box(body, f"{name}_top_col", (-sx / 2, -sy / 2, zt - tt, sx / 2, sy / 2, zt))
        B.static(M.table_top((sx, sy), tt), top_m, (cx, cy, zt))
        if t["kind"] == "two_top":
            body.add_geom(name=f"{name}_column_col", type=G.mjGEOM_CYLINDER, size=[0.045, (zt - tt - 0.04) / 2, 0],
                          pos=[0, 0, 0.04 + (zt - tt - 0.04) / 2], **STATIC)
            for k, R in enumerate((np.eye(3), M.Rz(math.pi / 2))):
                body.add_geom(name=f"{name}_base{k}_col", type=G.mjGEOM_BOX, size=[0.24, 0.03, 0.025],
                              pos=[0, 0, 0.025], quat=list(_quat(R)), **STATIC)
            B.static(M.pedestal(zt - tt), iron, (cx, cy, 0.0))
        else:
            ins = RM.LEG["inset"]
            for i, x in enumerate((-sx / 2 + ins, sx / 2 - ins)):
                for j, y in enumerate((-sy / 2 + ins, sy / 2 - ins)):
                    collider_box(body, f"{name}_leg{i}{j}_col", (x - 0.025, y - 0.025, 0.0, x + 0.025, y + 0.025,
                                                                 zt - tt))
                    B.static(M.tapered_leg(zt - tt), iron, (cx + x, cy + y, 0.0))
            for sgn in (-1, 1):
                collider_box(body, f"{name}_apron{'pn'[sgn > 0]}y_col",
                             (-sx / 2 + ins, sgn * (sy / 2 - ins) - 0.01, zt - tt - 0.07, sx / 2 - ins,
                              sgn * (sy / 2 - ins) + 0.01, zt - tt))
                slab(B, (cx - sx / 2 + ins, cy + sgn * (sy / 2 - ins) - 0.01, zt - tt - 0.07, cx + sx / 2 - ins,
                         cy + sgn * (sy / 2 - ins) + 0.01, zt - tt), iron)
        for st in t["seats"]:
            x, y = st["chair"]
            yaw = st["facing"]
            R = M.Rz(yaw)
            B.static(frame, bent, (x, y, 0.0), R)
            B.static(seat, cane, (x, y, 0.0), R)
            cb = wb.add_body(name=f"chair_{st['id']}", pos=[x, y, 0.0], quat=quat_z(yaw))
            cz = RM.CHAIR["seat_z"]
            cb.add_geom(name=f"chair_{st['id']}_seat_col", type=G.mjGEOM_CYLINDER, size=[0.20, 0.015, 0],
                        pos=[0, 0, cz - 0.015], **STATIC)
            cb.add_geom(name=f"chair_{st['id']}_back_col", type=G.mjGEOM_BOX, size=[0.02, 0.15, 0.20],
                        pos=[-0.205, 0, RM.CHAIR["back_h"] - 0.21], **STATIC)
            for k, (lx, ly) in enumerate(((0.16, 0.15), (0.16, -0.15), (-0.19, 0.15), (-0.19, -0.15))):
                cb.add_geom(name=f"chair_{st['id']}_leg{k}_col", type=G.mjGEOM_CAPSULE, size=[0.014, cz / 2 - 0.02, 0],
                            pos=[lx, ly, cz / 2 - 0.01], **STATIC)
        # pendants over the table (drawn) and their lights
        xs = (cx,) if t["kind"] == "two_top" else (cx - 0.25, cx + 0.25)
        for k, xp in enumerate(xs):
            B.static(pend, brass, (xp, cy, RM.WALL_H))
            B.static(pend_in, shade_in, (xp, cy, RM.WALL_H))
            B.static(pend_bulb, bulb, (xp, cy, RM.WALL_H))
            lt = wb.add_light(name=f"pendant_{name}{k}", type=mujoco.mjtLightType.mjLIGHT_SPOT,
                              pos=[xp, cy, RM.WALL_H - 1.05], dir=[0.0, 0.0, -1.0])
            lt.cutoff, lt.range, lt.castshadow, lt.bulbradius = 62.0, 9.0, False, 0.12


def add_station(B: O.Builder):
    """The service station on the east wall, the dish-return pass on the north wall, the host stand, the planters,
    the mop bucket, the floor tape of the dock and the cone's spot, the wall decor."""
    wb = B.s.worldbody
    walnut = B.material("walnut", tex="walnut", tile=0.9, rough=0.42)
    oak = B.material("oak", tex="oak", tile=0.9, rough=0.5)
    stainless = B.material("stainless", tex="stainless", tile=0.5, rough=0.35, metal=1.0)
    porcelain = B.material("porcelain", (0.96, 0.96, 0.94, 1), rough=0.22)
    glass = B.material("glass", (0.88, 0.94, 0.97, 0.22), rough=0.05)
    dark = B.material("trim_dark", (0.10, 0.14, 0.12, 1), rough=0.5)
    brass = B.material("brass", (0.80, 0.62, 0.32, 1), rough=0.3, metal=1.0)
    x0, y0, z0, x1, y1, z1 = RM.STATION["box"]
    collider_box(wb, "station_col", (x0, y0, 0.0, x1, y1, z1))
    slab(B, (x0 + 0.02, y0 + 0.02, 0.0, x1, y1 - 0.02, z1 - 0.03), walnut, 0.004)
    slab(B, (x0 - 0.02, y0, z1 - 0.03, x1, y1, z1), oak, 0.006)
    for k in range(4):                                       # door seams and knobs
        yy = y0 + (y1 - y0) * (k + 0.5) / 4
        slab(B, (x0 + 0.015, yy - 0.004, 0.10, x0 + 0.02, yy + 0.004, z1 - 0.06), dark)
        B.static(P.transform(P.lathe([(0.0, 0.0), (0.012, 0.0), (0.012, 0.02), (0.0, 0.02)], 12), R=M.Ry(-math.pi / 2),
                             t=(x0 + 0.02, yy + 0.06, 0.70)), brass)
    for k, zs in enumerate(RM.STATION["shelves_z"]):
        sb = (x1 - RM.STATION["shelf_depth"], y0, zs - 0.025, x1, y1, zs)
        collider_box(wb, f"station_shelf{k}_col", sb)
        slab(B, sb, oak, 0.005)
    # what stands on the shelves and the counter (drawn): plate stacks, upturned glasses, a carafe, menus, the POS
    for k, yy in enumerate((y0 + 0.30, y0 + 0.62)):
        B.static(M.plate_stack(8), porcelain, (x1 - 0.16, yy, RM.STATION["shelves_z"][0]))
    B.static(M.glass_rows(2, 5, 0.085), glass, (x1 - 0.13, y0 + 1.45, RM.STATION["shelves_z"][0]), M.Rz(math.pi / 2))
    B.static(M.glass_rows(2, 6, 0.085), glass, (x1 - 0.13, y0 + 1.05, RM.STATION["shelves_z"][1]), M.Rz(math.pi / 2))
    carafe = P.lathe([(0.05, 0.0), (0.055, 0.12), (0.035, 0.20), (0.032, 0.25), (0.0, 0.25)], 24)
    B.static(carafe, glass, (x1 - 0.20, y1 - 0.25, z1))
    slab(B, (x1 - 0.30, y0 + 0.08, z1, x1 - 0.06, y0 + 0.40, z1 + 0.04), B.material("menus", (0.18, 0.14, 0.10, 1),
                                                                               rough=0.6))
    pos_screen = B.material("pos_screen", (0.10, 0.20, 0.28, 1), rough=0.2, emission=0.6)
    slab(B, (x1 - 0.22, y0 + 1.70, z1, x1 - 0.14, y0 + 1.94, z1 + 0.20), B.material("pos_body", (0.15, 0.15, 0.16, 1),
                                                                              rough=0.4))
    slab(B, (x1 - 0.225, y0 + 1.72, z1 + 0.03, x1 - 0.22, y0 + 1.92, z1 + 0.18), pos_screen)
    # the clean roll-ups' wire baskets on the counter
    wire = B.material("basket_wire", (0.12, 0.12, 0.13, 1), rough=0.5, metal=0.6)
    for k, bk in enumerate(RM.BASKETS):
        (bxc, byc), (bsx, bsy), bh = bk["center"], bk["size"], bk["h"]
        z0b = z1
        wb.add_geom(name=f"basket{k}_floor_col", type=G.mjGEOM_BOX, size=[bsx / 2, bsy / 2, 0.003],
                    pos=[bxc, byc, z0b + 0.003], **STATIC)
        for sgn in (-1, 1):
            wb.add_geom(name=f"basket{k}_wall{'pn'[sgn > 0]}x_col", type=G.mjGEOM_BOX, size=[0.003, bsy / 2, bh / 2],
                        pos=[bxc + sgn * (bsx / 2 - 0.003), byc, z0b + bh / 2], **STATIC)
            wb.add_geom(name=f"basket{k}_wall{'pn'[sgn > 0]}y_col", type=G.mjGEOM_BOX, size=[bsx / 2, 0.003, bh / 2],
                        pos=[bxc, byc + sgn * (bsy / 2 - 0.003), z0b + bh / 2], **STATIC)
        B.static(M.open_box(bsx, bsy, bh, 0.006, 0.002), wire, (bxc, byc, z0b))
    # the dish-return pass: a stainless counter under the opening, the dish room behind it
    px0, py0, pz0, px1, py1, pz1 = RM.PASS["counter"]
    collider_box(wb, "pass_col", (px0, py0, 0.0, px1, py1 + 0.6, pz1))
    slab(B, (px0, py0, pz1 - 0.03, px1, py1 + 0.6, pz1), stainless, 0.004)
    slab(B, (px0 + 0.02, py0 + 0.02, 0.0, px1 - 0.02, py1, pz1 - 0.03), stainless)
    kit = B.material("kitchen_tiles", tex="kitchen", tile=1.2, rough=0.4)
    oz0, oz1 = RM.PASS["opening_z"]
    Y = RM.ROOM_Y
    slab(B, (px0 - 0.4, Y + 1.1, 0.0, px1 + 0.4, Y + 1.15, RM.WALL_H), kit)
    slab(B, (px0 - 0.4, Y + RM.WALL_T, -0.01, px1 + 0.4, Y + 1.15, 0.0), B.material("quarry", (0.45, 0.30, 0.24, 1),
                                                                                 rough=0.8))
    slab(B, (px0, Y + 0.85, 1.30, px1, Y + 1.10, 1.33), stainless)
    for xw in (px0 - 0.45, px1 + 0.40):                       # the dish room's side walls and ceiling
        slab(B, (xw, Y + RM.WALL_T, 0.0, xw + 0.05, Y + 1.15, RM.WALL_H), kit)
    slab(B, (px0 - 0.45, Y + RM.WALL_T, 2.45, px1 + 0.45, Y + 1.15, 2.50), B.material("kitchen_ceiling", (0.85, 0.86,
                                                                                         0.86, 1), rough=0.8))
    slab(B, (px0 + 0.3, Y + 0.6, 2.42, px1 - 0.3, Y + 0.75, 2.45), B.material("kitchen_tube", (1.0, 1.0, 1.0, 1),
                                                                             rough=0.5, emission=2.0))
    for k in range(5):
        B.static(M.plate_stack(6, 0.11), porcelain, (px0 + 0.25 + k * 0.38, Y + 0.97, 1.33))
    lamp = B.material("heat_lamp", (1.0, 0.45, 0.15, 1), rough=0.4, emission=2.5)
    slab(B, (px0 + 0.1, py0 + 0.20, oz1 - 0.12, px1 - 0.1, py0 + 0.32, oz1 - 0.06), stainless)
    slab(B, (px0 + 0.15, py0 + 0.22, oz1 - 0.125, px1 - 0.15, py0 + 0.30, oz1 - 0.12), lamp)
    sign = B.material("sign_dish", tex="sign_dish", rough=0.6)
    B.static(P.transform(P.quad(0.80, 0.15, "-y"), t=((px0 + px1) / 2, Y - 0.006, oz1 + 0.15)), sign)
    # the dock's floor tape (yellow) and the cone's spot
    tape = B.material("tape", (0.95, 0.78, 0.10, 1), rough=0.6)
    dx0, dy0, dx1, dy1 = RM.DOCK["box"]
    for box in ((dx0, dy0, dx1, dy0 + 0.05), (dx0, dy1 - 0.05, dx1, dy1), (dx0, dy0, dx0 + 0.05, dy1),
                (dx1 - 0.05, dy0, dx1, dy1)):
        slab(B, (box[0], box[1], 0.0, box[2], box[3], 0.0015), tape)
    cx, cy = RM.CONE_HOME
    for box in ((cx - 0.22, cy - 0.22, cx + 0.22, cy - 0.19), (cx - 0.22, cy + 0.19, cx + 0.22, cy + 0.22),
                (cx - 0.22, cy - 0.22, cx - 0.19, cy + 0.22), (cx + 0.19, cy - 0.22, cx + 0.22, cy + 0.22)):
        slab(B, (box[0], box[1], 0.0, box[2], box[3], 0.0015), tape)
    # the host stand
    hb = RM.HOST_STAND["box"]
    collider_box(wb, "host_col", hb)
    slab(B, (hb[0], hb[1], 0.0, hb[3], hb[4], hb[5] - 0.04), walnut, 0.01)
    slab(B, (hb[0] - 0.03, hb[1] - 0.05, hb[5] - 0.04, hb[3] + 0.03, hb[4] + 0.02, hb[5]), oak, 0.008)
    # planters (fixed): pot colliders, drawn pots and foliage
    leaf = B.material("leaves", (0.24, 0.40, 0.20, 1), rough=0.7)
    trunk = B.material("trunk", (0.30, 0.22, 0.15, 1), rough=0.8)
    pot_m = B.material("pot", tex="pot_ceramic", tile=0.5, rough=0.35)
    soil = B.material("soil", (0.16, 0.11, 0.08, 1), rough=1.0)
    for k, (pn, pl) in enumerate(sorted(RM.PLANTERS.items())):
        (qx, qy), r, h = pl["center"], pl["radius"], pl["h"]
        wb.add_geom(name=f"planter_{pn}_col", type=G.mjGEOM_CYLINDER, size=[r, h / 2, 0], pos=[qx, qy, h / 2],
                    **STATIC)
        pot, sl = M.planter(r, h)
        B.static(pot, pot_m, (qx, qy, 0.0))
        B.static(sl, soil, (qx, qy, 0.0))
        lv, tr = M.foliage((qx, qy), pl["foliage_r"], pl["foliage_z"][0], pl["foliage_z"][1], 40 + k, n=900,
                           leaf=(0.11, 0.045))
        B.static(lv, leaf)
        B.static(tr, trunk)
    # the mop bucket (fixed, drawn) and the mop's wall clip
    mb = RM.MOP_HOME["bucket"]
    bx, by, bz = RM.MOP_HOME["bucket_size"]
    collider_box(wb, "mop_bucket_col", (mb[0] - bx / 2, mb[1] - by / 2, 0.0, mb[0] + bx / 2, mb[1] + by / 2, bz))
    body, water, wringer, wheels = M.mop_bucket()
    yel = B.material("mop_bucket_yellow", (0.95, 0.74, 0.08, 1), rough=0.5)
    B.static(body, yel, (mb[0], mb[1], 0.0))
    B.static(water, B.material("mop_water", (0.45, 0.52, 0.50, 1), rough=0.1), (mb[0], mb[1], 0.0))
    B.static(wringer, B.material("wringer_grey", (0.30, 0.31, 0.33, 1), rough=0.5), (mb[0], mb[1], 0.0))
    B.static(wheels, B.material("cart_wheel", (0.05, 0.05, 0.05, 1), rough=0.6), (mb[0], mb[1], 0.0))
    cxp, cyp, czp = RM.MOP_HOME["clip"]
    slab(B, (cxp - 0.03, cyp - 0.035, czp - 0.03, RM.ROOM_X, cyp + 0.035, czp + 0.03), B.material(
        "clip_black", (0.08, 0.08, 0.09, 1), rough=0.5), 0.006)
    # wall decor: the menu board, framed art, the name, sconces
    chalk = B.material("chalkboard", tex="chalkboard", rough=0.85)
    frame_m = B.material("frame_black", (0.06, 0.05, 0.05, 1), rough=0.45)
    fr, cv = M.frame_picture(1.30, 0.98)
    R = M.Rz(-math.pi / 2)                                   # +y facing -> +x facing (the west wall)
    B.static(fr, frame_m, (0.0, 4.15, 1.78), R)
    B.static(cv, chalk, (0.0, 4.15, 1.78), R)
    for (yy, w_, h_, tex) in ((1.70, 0.62, 0.78, "art_field"), (6.60, 0.80, 0.64, "art_landscape")):
        fr2, cv2 = M.frame_picture(w_, h_)
        B.static(fr2, frame_m, (0.0, yy, 1.85), R)
        B.static(cv2, B.material(f"art_{tex}", tex=tex, rough=0.8), (0.0, yy, 1.85), R)
    fr3, cv3 = M.frame_picture(0.52, 0.66)
    B.static(fr3, frame_m, (5.0, RM.ROOM_Y, 1.90), M.Rz(math.pi))
    B.static(cv3, B.material("art_botanical", tex="art_botanical", rough=0.8), (5.0, RM.ROOM_Y, 1.90), M.Rz(math.pi))
    name = B.material("sign_linden", tex="sign_linden", rough=0.6)
    B.static(P.transform(P.quad(1.40, 0.35, "-y"), t=(2.6, RM.ROOM_Y - 0.006, 2.55)), name)
    globe = B.material("sconce_globe", (1.0, 0.94, 0.84, 1), rough=0.3, emission=2.0)
    for yy in (2.90, 5.45):
        slab(B, (0.0, yy - 0.05, 1.98, 0.03, yy + 0.05, 2.12), brass, 0.01)
        B.static(P.transform(P.lathe([(0.0, -0.06), (0.05, -0.04), (0.06, 0.0), (0.05, 0.04), (0.0, 0.06)], 20),
                             t=(0.11, yy, 2.12)), globe)
        slab(B, (0.02, yy - 0.01, 2.05, 0.11, yy + 0.01, 2.07), brass)


# ================================================================================================ build
def build(layout: dict, *, offscreen=(960, 720)):
    """The cell spec for one episode layout (layout.draw): returns (spec, meta). meta: {"features": {body: [grasp
    features]}, "graspable": [body names], "cart": the cart's record, "soil": cells, "spill": cells, "crumbs": n,
    "guests": {seat: body}, ...}."""
    s = mujoco.MjSpec()
    s.modelname = "bistro_cell"
    s.compiler.degree = False
    o = s.option
    o.timestep = 0.002
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.iterations = 60
    o.ls_iterations = 30
    s.memory = 512 * 1024 * 1024
    s.visual.global_.offwidth, s.visual.global_.offheight = offscreen
    s.visual.quality.shadowsize = 2048
    s.visual.headlight.ambient = [0.30, 0.30, 0.30]
    s.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    s.visual.headlight.specular = [0.0, 0.0, 0.0]
    s.visual.map.znear = 0.01
    s.stat.extent, s.stat.center = 12.0, [RM.ROOM_X / 2, RM.ROOM_Y / 2, 1.2]
    numeric, text = filament_view_keys()
    for key, value in sorted(numeric.items()):
        s.add_numeric(name=key, data=[value])
    for key, value in sorted(text.items()):
        s.add_text(name=key, data=value)
    B = O.Builder(s)
    wb = s.worldbody
    sun = wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[5.0, -6.0, 8.0],
                       dir=list(np.asarray(SUN_DIR) / np.linalg.norm(SUN_DIR)))
    sun.castshadow = True
    for k, (xl, yl) in enumerate(((7.8, 3.9), (8.6, 6.6))):
        lt = wb.add_light(name=f"service_{k}", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=[xl, yl, RM.WALL_H - 0.05],
                          dir=[0.0, 0.0, -1.0])
        lt.cutoff, lt.range, lt.castshadow, lt.bulbradius = 70.0, 10.0, False, 0.2
    for k, (xl, yl) in enumerate(((1.5, 2.4), (4.2, 2.4), (6.9, 2.4), (2.0, 5.7), (5.0, 5.7), (8.2, 5.8), (8.6, 2.0))):
        lt = wb.add_light(name=f"bounce_{k}", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=[xl, yl, 0.05],
                          dir=[0.0, 0.0, 1.0])
        lt.cutoff, lt.range, lt.castshadow, lt.bulbradius = 85.0, 8.0, False, 0.6
    add_shell(B)
    add_tables(B, layout)
    add_station(B)
    meta = {"features": {}, "graspable": [], "guests": {}, "kinds": {}}
    # the bus cart
    cart = O.add_cart(B, wb, layout["cart"])
    meta["cart"] = cart
    meta["features"]["cart"] = cart["features"]
    meta["graspable"].append("cart")
    meta["kinds"]["cart"] = "cart"
    # the guests seated at the start
    for seat_id, look in sorted(layout["guests"].items()):
        seat = RM_SEATS()[seat_id]
        O.add_guest(B, wb, seat, look, f"guest_{seat_id}")
        meta["guests"][seat_id] = f"guest_{seat_id}"
    # the loose objects (free bodies at the layout's poses)
    for name, (kind, params) in sorted(layout["loose"].items()):
        pos, quat = layout["objects"][name]
        body = wb.add_body(name=name, pos=list(pos), quat=list(quat))
        body.add_freejoint(name=f"{name}_free")
        feats = O.BUILDERS[kind](B, body, name, params)
        meta["features"][name] = feats
        meta["graspable"].append(name)
        meta["kinds"][name] = kind
    # crumbs: drawn-only mocap bodies (the cell moves them; cleaning.Crumbs)
    rng = np.random.default_rng(int(layout.get("look_seed", 0)))
    for i, p in enumerate(layout["crumbs"]):
        cb = wb.add_body(name=f"crumb{i}", pos=list(p), mocap=True)
        r = float(layout["crumb_r"][i])
        col = CRUMB_COLOURS[int(rng.integers(len(CRUMB_COLOURS)))]
        sh = rng.uniform(0.6, 1.0, size=3)
        cb.add_geom(name=f"crumb{i}_vis", type=G.mjGEOM_ELLIPSOID, size=[r * sh[0], r * sh[1], r * 0.55 * sh[2]],
                    rgba=list(col) + [1.0], contype=0, conaffinity=0, group=1, mass=0.0,
                    quat=quat_z(rng.uniform(0, math.pi)))
    meta["crumbs"] = len(layout["crumbs"])
    # the soiling of the tables (food smears, drink rings): drawn-only cells the cell colours
    soil = layout["soil"]
    sbody = wb.add_body(name="soil")
    c = soil["cell"]
    for k, (tname, x, y, g0) in enumerate(soil["cells"]):
        sbody.add_geom(name=f"soil{k}", type=G.mjGEOM_BOX, size=[c / 2, c / 2, 0.0002],
                       pos=[x, y, RM.TABLE_TOP_Z + 0.0004], rgba=list(SOIL_RGB) + [0.0], contype=0, conaffinity=0,
                       group=1, mass=0.0)
    meta["soil"] = soil
    # the floor spill (drawn-only cells, invisible until it happens)
    sp = layout["spill"]
    pbody = wb.add_body(name="spill")
    c = sp["cell"]
    rgb = SPILL_RGB[sp["drink"]]
    for k, (x, y) in enumerate(sp["cells"]):
        pbody.add_geom(name=f"spill{k}", type=G.mjGEOM_BOX, size=[c / 2 + 0.001, c / 2 + 0.001, 0.0003],
                       pos=[x, y, 0.0006], rgba=list(rgb) + [0.0], contype=0, conaffinity=0, group=1, mass=0.0)
    meta["spill"] = sp
    # the static decor, merged per material
    B.flush(wb)
    # the robot
    frame = wb.add_frame(name="robot_frame", pos=[0.0, 0.0, 0.0])
    s.attach(g1_spec(ROBOT_NAME), prefix=f"{ROBOT_NAME}/", frame=frame)
    # ceiling cameras
    for cname, cam in RM.CEILING_CAMERAS.items():
        wb.add_camera(name=cname, pos=list(cam["pos"]), quat=list(_look_at_quat(cam["pos"], cam["target"])),
                      fovy=cam["fovy"])
    # inactive weld pool: every hand x every graspable loose body (grips; the cart is held by the cell's own servo),
    # the mop's wall clip and the crumb pan's hook on the cart
    weld = dict(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, active=False,
                solref=[0.01, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
                data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1.0])
    for side in ("left", "right"):
        for name in meta["graspable"]:
            if name == "cart":
                continue
            s.add_equality(name=f"grip_{side}_{name}", name1=f"{ROBOT_NAME}/{side}_wrist_yaw_link", name2=name,
                           **weld)
    if "mop" in layout["loose"]:
        s.add_equality(name="clip_mop", name1="world", name2="mop", **weld)
    if "crumb_pan" in layout["loose"]:
        s.add_equality(name="clip_crumb_pan", name1="cart", name2="crumb_pan", **weld)
    return s, meta


def RM_SEATS():
    return {st["id"]: st for t in RM.tables().values() for st in t["seats"]}
