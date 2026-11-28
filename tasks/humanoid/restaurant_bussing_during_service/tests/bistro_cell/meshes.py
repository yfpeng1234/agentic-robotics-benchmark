"""What the cameras see of the bistro's furniture, tableware and equipment: procedural meshes built from props.py's
primitives (deterministic; metres; each in its object's own frame). Colliders are primitive shapes (objects.py,
scene.py)."""

from __future__ import annotations

import math

import numpy as np

from . import props as P


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def planar_uv(mesh, scale=1.0, axes=(0, 1), offset=(0.0, 0.0)):
    """Replace a mesh's texture coordinates by a planar projection (u, v) = coords[axes] / scale + offset."""
    V, F, N, UV = mesh
    uv = np.stack([V[:, axes[0]] / scale + offset[0], V[:, axes[1]] / scale + offset[1]], axis=1)
    return V, F, N, uv


def _ring(r, z, n=48):
    a = np.linspace(0.0, 2 * math.pi, n + 1)
    return np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, z)], axis=1)


# ================================================================================================ furniture
def bentwood_chair(seat_z=0.46, seat_r=0.205, back_h=0.90):
    """A bentwood bistro chair in its frame (origin on the floor under the seat's middle, +x the way a sitter faces):
    (frame, seat). The frame is the bent beech: front legs, rear legs that rise into the back, the back's bow and inner
    loop, the seat ring and the leg ring; the seat is a cane disc (planar texture coordinates, 1 unit = 0.4 m)."""
    rt = 0.0125
    parts = [P.tube(_ring(seat_r + 0.004, seat_z - 0.013, 56), 0.0145, segments=10, cap=False)]
    for s in (-1.0, 1.0):
        a = math.radians(42.0) * s
        top = np.array([0.165 * math.cos(a), 0.165 * math.sin(a), seat_z - 0.02])
        bot = np.array([0.215 * math.cos(a), 0.215 * math.sin(a), 0.0])
        parts.append(P.tube(np.linspace(top, bot, 6), rt, segments=10))
        pts = [(-0.215, 0.168 * s, 0.0), (-0.168, 0.138 * s, seat_z - 0.02), (-0.172, 0.132 * s, seat_z + 0.16),
               (-0.198, 0.126 * s, back_h - 0.13), (-0.214, 0.095 * s, back_h - 0.012)]
        parts.append(P.tube(P.catmull_rom(pts, 6), rt, segments=10))
    bow = [(-0.214, -0.095, back_h - 0.012), (-0.221, -0.05, back_h + 0.001), (-0.223, 0.0, back_h + 0.004),
           (-0.221, 0.05, back_h + 0.001), (-0.214, 0.095, back_h - 0.012)]
    parts.append(P.tube(P.catmull_rom(bow, 6), rt, segments=10, cap=False))
    loop = [(-0.176, -0.095, seat_z + 0.12), (-0.192, -0.085, back_h - 0.20), (-0.200, -0.045, back_h - 0.105),
            (-0.202, 0.0, back_h - 0.09), (-0.200, 0.045, back_h - 0.105), (-0.192, 0.085, back_h - 0.20),
            (-0.176, 0.095, seat_z + 0.12)]
    parts.append(P.tube(P.catmull_rom(loop, 6), 0.0095, segments=8))
    parts.append(P.tube(_ring(0.188, 0.21, 48), 0.009, segments=8, cap=False))
    frame = P.merge(*parts)
    seat = P.lathe([(0.0, seat_z - 0.012), (seat_r - 0.004, seat_z - 0.012), (seat_r, seat_z - 0.004),
                    (seat_r - 0.006, seat_z + 0.004), (0.0, seat_z + 0.006)], segments=48)
    return frame, planar_uv(seat, 0.40, offset=(0.5, 0.5))


def table_top(size, t, radius=0.008):
    """A table top (centred on its middle, z from -t to 0): a rounded slab; texture coordinates 1 unit = 1 m, the
    grain along x."""
    sx, sy = size
    m = P.rounded_box([sx / 2, sy / 2, t / 2], radius, 4)
    return P.transform(m, t=(0.0, 0.0, -t / 2))


def pedestal(top_z, column_r=0.040, base_half=0.24, base_h=0.035):
    """A cast-iron pedestal (origin on the floor): a cross base with four pads, a turned column and the spider plate
    under the top."""
    parts = []
    for R in (np.eye(3), Rz(math.pi / 2)):
        arm = P.rounded_box([base_half, 0.026, base_h / 2], 0.010, 3)
        parts.append(P.transform(arm, R=R, t=(0.0, 0.0, base_h / 2 + 0.008)))
    for a in range(4):
        x, y = base_half * 0.93 * math.cos(a * math.pi / 2), base_half * 0.93 * math.sin(a * math.pi / 2)
        parts.append(P.transform(P.lathe([(0.0, 0.0), (0.030, 0.0), (0.026, 0.010), (0.0, 0.010)], 16),
                                 t=(x, y, 0.0)))
    prof = [(0.075, base_h + 0.008), (0.060, base_h + 0.030), (0.045, base_h + 0.050), (column_r, base_h + 0.08),
            (column_r * 0.85, 0.30), (column_r, 0.45), (column_r * 0.85, top_z - 0.12), (0.055, top_z - 0.06),
            (0.055, top_z - 0.045)]
    parts.append(P.lathe(prof, 24))
    parts.append(P.transform(P.rounded_box([0.17, 0.17, 0.008], 0.004, 2), t=(0.0, 0.0, top_z - 0.045)))
    return P.merge(*parts)


def tapered_leg(h, top=0.05, bottom=0.032):
    """A tapered square leg (origin at its foot)."""
    n = 6
    parts = []
    for i in range(n):
        z0, z1 = h * i / n, h * (i + 1) / n
        w = bottom + (top - bottom) * (i + 0.5) / n
        parts.append(P.transform(P.box([w / 2, w / 2, (z1 - z0) / 2]), t=(0.0, 0.0, (z0 + z1) / 2)))
    return P.merge(*parts)


# ================================================================================================ tableware
PLATE_PROFILE = [(0.0, 0.006), (0.072, 0.006), (0.076, 0.0), (0.082, 0.0), (0.086, 0.006), (0.108, 0.015),
                 (0.120, 0.021), (0.116, 0.022), (0.103, 0.0185), (0.080, 0.0095), (0.0, 0.0095)]


def plate():
    return P.lathe(PLATE_PROFILE, segments=48), P.lathe([(0.110, 0.0195), (0.1165, 0.0212)], segments=48)


def tumbler(r=0.035, h=0.115):
    outer = [(0.030, 0.0), (0.031, 0.012), (r, h)]
    inner = [(r - 0.003, h), (0.028, 0.016), (0.0, 0.016)]
    return P.lathe(outer + inner, segments=36, cap_bottom=True)


def liquid(level, r_bottom=0.0275, r_top=0.0315, z0=0.0165):
    """The drink left in a tumbler: a frustum from the glass's inner floor up to `level` (m above the glass's foot)."""
    rt = r_bottom + (r_top - r_bottom) * (level - z0) / (0.115 - z0)
    return P.lathe([(0.0, z0), (r_bottom, z0), (rt, level), (0.0, level)], segments=28)


def fork():
    """A table fork lying flat (origin at its middle, its axis along x, the tines towards +x, its underside at z=0)."""
    parts = [P.transform(P.rounded_box([0.052, 0.0085, 0.0025], 0.0022, 2), t=(-0.045, 0.0, 0.0035)),
             P.transform(P.rounded_box([0.020, 0.0045, 0.0015], 0.0012, 2), t=(0.022, 0.0, 0.0045)),
             P.transform(P.rounded_box([0.012, 0.0115, 0.0013], 0.0010, 2), t=(0.048, 0.0, 0.0055))]
    for k in range(4):
        y = -0.0090 + k * 0.0060
        parts.append(P.transform(P.rounded_box([0.020, 0.0016, 0.0011], 0.0008, 1), t=(0.078, y, 0.0068)))
    return P.merge(*parts)


def knife():
    """A table knife lying flat (origin at its middle, the blade towards +x, its underside at z=0)."""
    blade = [(0.0, -0.0075), (0.072, -0.0072), (0.090, -0.0045), (0.098, 0.0), (0.090, 0.0080), (0.0, 0.0090)]
    parts = [P.transform(P.rounded_box([0.050, 0.0085, 0.0040], 0.0035, 2), t=(-0.050, 0.0, 0.0042)),
             P.transform(P.extrude(blade, 0.0018, axis="z"), t=(0.004, 0.0, 0.0030))]
    return P.merge(*parts)


def rollup(length=0.23, r=0.021):
    """A roll-up (a linen napkin rolled round a knife and a fork, a paper band round its middle): its axis along x,
    centred, resting on z = 0. Returns (napkin, band)."""
    body = P.lathe([(0.0, -length / 2), (r - 0.004, -length / 2), (r, -length / 2 + 0.006), (r, length / 2 - 0.006),
                    (r - 0.004, length / 2), (0.0, length / 2)], segments=24)
    band = P.lathe([(r + 0.0006, -0.022), (r + 0.0006, 0.022)], segments=24)
    R = Ry(math.pi / 2)
    return P.transform(body, R=R, t=(0, 0, r)), P.transform(band, R=R, t=(0, 0, r))


def napkin(seed, r=0.040):
    """A used linen napkin dropped in a loose heap (a flattened crumpled ball)."""
    V, F, N, UV = P.crumpled_ball(r, seed, subdiv=3, roughness=0.30)
    V = V * np.array([1.15, 0.95, 0.55])
    V[:, 2] -= V[:, 2].min()
    return V, F, N, UV


def food(kind, seed):
    """Leftovers on a plate (drawn only): a list of (mesh, material key) in the plate's frame (z = the plate's well)."""
    rng = np.random.default_rng(seed)
    out = []
    z = 0.0096
    # a sauce smear
    a = rng.uniform(0, 2 * math.pi)
    rx, ry = rng.uniform(0.035, 0.055), rng.uniform(0.018, 0.03)
    cx, cy = rng.uniform(-0.03, 0.03, size=2)
    sm = P.lathe([(0.0, 0.0), (1.0, 0.0), (0.92, 0.0012), (0.0, 0.0016)], segments=24)
    sm = (sm[0] * np.array([rx, ry, 1.0]), sm[1], sm[2], sm[3])
    out.append((P.transform(sm, R=Rz(a), t=(cx, cy, z)), {"pasta": "sauce_red", "steak": "sauce_brown",
                                                          "salad": "dressing", "fish": "sauce_cream"}[kind]))
    if kind == "pasta":
        for i in range(int(rng.integers(5, 9))):
            p0 = rng.uniform(-0.03, 0.03, size=2)
            pts = [(p0[0] + 0.02 * math.cos(t), p0[1] + 0.015 * math.sin(1.7 * t), z + 0.004 + 0.002 * math.sin(3 * t))
                   for t in np.linspace(0, rng.uniform(2.5, 4.5), 9)]
            out.append((P.tube(np.array(pts), 0.0021, segments=6), "pasta"))
    elif kind == "steak":
        for i in range(int(rng.integers(4, 8))):
            L = rng.uniform(0.035, 0.06)
            fr = P.rounded_box([L / 2, 0.0045, 0.0045], 0.0015, 1)
            out.append((P.transform(fr, R=Rz(rng.uniform(0, math.pi)), t=(rng.uniform(-0.05, 0.05),
                                                                         rng.uniform(-0.05, 0.05), z + 0.0045)), "fries"))
        fat = P.crumpled_ball(0.014, int(rng.integers(1 << 30)), subdiv=2, roughness=0.25)
        out.append((P.transform((fat[0] * np.array([1.4, 1.0, 0.45]), fat[1], fat[2], fat[3]),
                                t=(rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), z + 0.003)), "meat"))
    elif kind == "salad":
        for i in range(int(rng.integers(4, 7))):
            lf = P.lathe([(0.0, 0.0), (1.0, 0.0), (0.0, 0.0008)], segments=10)
            lf = (lf[0] * np.array([rng.uniform(0.012, 0.022), rng.uniform(0.008, 0.014), 1.0]), lf[1], lf[2], lf[3])
            out.append((P.transform(lf, R=Rz(rng.uniform(0, math.pi)) @ Rx(rng.uniform(-0.4, 0.4)),
                                    t=(rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), z + 0.003)), "greens"))
        tom = P.lathe([(0.0, -1.0), (0.7, -0.7), (1.0, 0.0), (0.7, 0.7), (0.0, 1.0)], segments=12)
        out.append((P.transform((tom[0] * 0.008, tom[1], tom[2], tom[3]),
                                t=(rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), z + 0.008)), "tomato"))
    else:   # fish: a lemon wedge and some greens
        wedge = P.lathe([(0.0, -1.0), (0.8, -0.5), (1.0, 0.0), (0.8, 0.5), (0.0, 1.0)], segments=10)
        wedge = (wedge[0] * np.array([0.012, 0.022, 0.010]), wedge[1], wedge[2], wedge[3])
        out.append((P.transform(wedge, R=Rz(rng.uniform(0, math.pi)), t=(rng.uniform(-0.04, 0.04),
                                                                         rng.uniform(-0.04, 0.04), z + 0.008)), "lemon"))
        skin = P.rounded_box([0.03, 0.012, 0.003], 0.002, 2)
        out.append((P.transform(skin, R=Rz(rng.uniform(0, math.pi)), t=(rng.uniform(-0.03, 0.03),
                                                                        rng.uniform(-0.03, 0.03), z + 0.003)), "fish"))
    return out


# ================================================================================================ equipment
def open_box(sx, sy, h, wt, radius=0.008, floor=True):
    """An open-topped box (a bus tub, a caddy): four walls of thickness wt and a floor, outer size sx x sy, height h;
    origin at the middle of its underside."""
    parts = []
    if floor:
        parts.append(P.transform(P.rounded_box([sx / 2, sy / 2, wt / 2], min(radius, wt / 2 - 1e-4), 2),
                                 t=(0.0, 0.0, wt / 2)))
    for sgn in (-1.0, 1.0):
        parts.append(P.transform(P.rounded_box([sx / 2, wt / 2, h / 2], min(radius, wt / 2 - 1e-4), 2),
                                 t=(0.0, sgn * (sy / 2 - wt / 2), h / 2)))
        parts.append(P.transform(P.rounded_box([wt / 2, sy / 2 - wt, h / 2], min(radius, wt / 2 - 1e-4), 2),
                                 t=(sgn * (sx / 2 - wt / 2), 0.0, h / 2)))
    return P.merge(*parts)


def tub(sx, sy, h, wt):
    """A grey bus tub: an open box with a rolled rim and two hand holds moulded into the short ends."""
    body = open_box(sx, sy, h, wt, 0.0025)
    rim = []
    for sgn in (-1.0, 1.0):
        rim.append(P.transform(P.rounded_box([sx / 2 + 0.006, 0.010, 0.006], 0.004, 2), t=(0, sgn * (sy / 2 - 0.004),
                                                                                         h - 0.004)))
        rim.append(P.transform(P.rounded_box([0.012, sy / 2, 0.008], 0.005, 2), t=(sgn * (sx / 2 - 0.002), 0, h - 0.006)))
    return P.merge(body, *rim)


def bucket(r=0.105, h=0.20, wt=0.005):
    """The red sanitizer bucket: a tapered open pail with a lip and a wire bail (origin at its foot)."""
    rb = r - 0.012
    outer = [(rb, 0.0), (r, h - 0.012), (r + 0.006, h - 0.008), (r + 0.006, h), (r - wt, h), (rb - wt, wt),
             (0.0, wt)]
    body = P.lathe(outer, segments=40)
    foot = P.lathe([(0.0, 0.0), (rb, 0.0), (rb, 0.0005), (0.0, 0.0005)], segments=40)
    bail = [(r + 0.01 * math.cos(t), 0.0, h - 0.03 + 0.11 * math.sin(t)) for t in np.linspace(0, math.pi, 13)]
    bail = P.transform(P.tube(np.array(bail), 0.0022, segments=6), R=Rz(math.pi / 2))
    return P.merge(body, foot), bail


def cart(c):
    """The bus cart in its frame (origin on the floor under the top shelf's middle, x forward): (shelves, frame,
    wheels, handle) meshes."""
    sx, sy = c["size"]
    shelves = []
    for z in (c["top_z"], c["lower_z"]):
        shelves.append(P.transform(P.rounded_box([sx / 2, sy / 2, 0.016], 0.012, 3), t=(0.0, 0.0, z - 0.016)))
        lip = c["lip"]
        for sgn in (-1.0, 1.0):
            shelves.append(P.transform(P.rounded_box([sx / 2, 0.012, lip / 2 + 0.008], 0.008, 2),
                                       t=(0.0, sgn * (sy / 2 - 0.012), z + lip / 2 - 0.008)))
            shelves.append(P.transform(P.rounded_box([0.012, sy / 2, lip / 2 + 0.008], 0.008, 2),
                                       t=(sgn * (sx / 2 - 0.012), 0.0, z + lip / 2 - 0.008)))
    frame = []
    px, py = sx / 2 - 0.035, sy / 2 - 0.035
    for x in (-px, px):
        for y in (-py, py):
            frame.append(P.tube(np.array([(x, y, 0.105), (x, y, c["top_z"] - 0.01)]), 0.016, segments=12))
    wheels = []
    for x in (-px, px):
        for y in (-py, py):
            wheels.append(P.transform(P.rounded_box([0.022, 0.018, 0.022], 0.006, 2), t=(x, y, 0.095)))
            wh = P.lathe([(0.0, -0.016), (c["caster_r"] - 0.006, -0.016), (c["caster_r"], -0.010),
                          (c["caster_r"], 0.010), (c["caster_r"] - 0.006, 0.016), (0.0, 0.016)], segments=20)
            wheels.append(P.transform(wh, R=Rx(math.pi / 2), t=(x - 0.012, y, c["caster_r"])))
    hz, hx, hh = c["handle_z"], c["handle_x"], c["handle_half"]
    path = [(-sx / 2 + 0.02, -hh, c["top_z"] - 0.01), (hx + 0.015, -hh, hz - 0.03), (hx, -hh + 0.03, hz),
            (hx, hh - 0.03, hz), (hx + 0.015, hh, hz - 0.03), (-sx / 2 + 0.02, hh, c["top_z"] - 0.01)]
    handle = P.tube(P.catmull_rom(path, 6), c["handle_r"], segments=12)
    grip = P.transform(P.lathe([(c["handle_r"] + 0.004, -0.13), (c["handle_r"] + 0.004, 0.13)], segments=16),
                       R=Rx(math.pi / 2), t=(hx, 0.0, hz))
    return P.merge(*shelves), P.merge(*frame), P.merge(*wheels), (handle, grip)


def crumb_pan(width=0.22, depth=0.16, wall=0.05, t=0.003):
    """A table crumb pan (a small dustpan): its frame's origin at the middle of its lip, x from the lip to the back
    wall, its floor on z = 0. Returns (pan, handle) meshes; the handle is a short tube rising backwards from the back
    wall."""
    parts = [P.transform(P.box([depth / 2, width / 2, t / 2]), t=(depth / 2, 0.0, t / 2)),
             P.transform(P.box([t / 2, width / 2, wall / 2]), t=(depth - t / 2, 0.0, wall / 2))]
    for sgn in (-1.0, 1.0):
        prof = [(0.0, 0.0), (depth, 0.0), (depth, wall), (0.02, 0.008), (0.0, 0.004)]
        side = P.extrude(prof, t, axis="y")
        parts.append(P.transform(side, t=(0.0, sgn * (width / 2 - t / 2), 0.0)))
    lip = P.transform(P.rounded_box([0.006, width / 2, 0.0025], 0.002, 1), t=(0.004, 0.0, 0.0025))
    hl = P.tube(np.array([(depth - 0.005, 0.0, wall * 0.7), (depth + 0.13 * math.cos(math.radians(28)), 0.0,
                                                               wall * 0.7 + 0.13 * math.sin(math.radians(28)))]),
                0.011, segments=12)
    return P.merge(*parts), P.merge(lip, hl)


def cone_body(h=0.82):
    """The yellow wet-floor cone (origin at the middle of its base): (shell without the label band, the label band,
    the base plate, the top handle)."""
    base = P.transform(P.rounded_box([0.15, 0.15, 0.015], 0.012, 3), t=(0.0, 0.0, 0.015))
    r0, r1 = 0.115, 0.028

    def r_at(z):
        return r0 + (r1 - r0) * (z - 0.03) / (h - 0.06 - 0.03)
    low = P.lathe([(r0, 0.03), (r_at(0.30), 0.30)], segments=40)
    band = P.lathe([(r_at(0.30), 0.30), (r_at(0.58), 0.58)], segments=40)
    _, F, N, UV = band
    V = band[0]
    UV = np.stack([np.arctan2(V[:, 1], V[:, 0]) / (2 * math.pi) % 1.0 * 2.0, (V[:, 2] - 0.30) / 0.28], axis=1)
    band = (V, F, N, UV)
    high = P.lathe([(r_at(0.58), 0.58), (r1, h - 0.06), (0.020, h - 0.045), (0.0, h - 0.04)], segments=40)
    stem = P.lathe([(0.016, h - 0.05), (0.016, h - 0.005), (0.0, h)], segments=16)
    bar = P.transform(P.lathe([(0.0, -0.065), (0.0115, -0.062), (0.0125, -0.055), (0.0125, 0.055), (0.0115, 0.062),
                               (0.0, 0.065)], segments=16), R=Rx(math.pi / 2), t=(0.0, 0.0, h + 0.0))
    return P.merge(low, high), band, base, P.merge(stem, bar)


def mop_parts(handle_len=1.40):
    """The flat mop in its handle's frame (origin at the swivel joint, the handle along +z): (handle, grip) and the
    head in the head's frame (origin at the swivel, the head below): (frame, pad)."""
    handle = P.lathe([(0.0, 0.02), (0.0125, 0.025), (0.0125, handle_len), (0.0, handle_len + 0.006)], segments=16)
    grip = P.lathe([(0.0145, handle_len - 0.20), (0.0145, handle_len - 0.01)], segments=16)
    yoke = P.merge(P.transform(P.rounded_box([0.018, 0.026, 0.020], 0.006, 2), t=(0.0, 0.0, 0.0)),
                   P.lathe([(0.0, 0.0), (0.016, 0.0), (0.016, 0.03), (0.0, 0.03)], segments=12))
    frame = P.transform(P.rounded_box([0.20, 0.055, 0.010], 0.006, 2), t=(0.0, 0.0, -0.022))
    pad = P.transform(P.rounded_box([0.21, 0.065, 0.007], 0.005, 2), t=(0.0, 0.0, -0.039))
    return (P.merge(handle, yoke), grip), (frame, pad)


def mop_bucket():
    """The yellow mop bucket with its side-press wringer (origin on the floor under its middle; drawn on the floor)."""
    body = P.transform(P.rounded_box([0.20, 0.14, 0.15], 0.03, 3), t=(0.0, 0.0, 0.18))
    water = P.transform(P.box([0.17, 0.11, 0.002]), t=(0.0, 0.0, 0.28))
    wringer = P.transform(P.rounded_box([0.09, 0.13, 0.08], 0.012, 2), t=(0.05, 0.0, 0.37))
    lever = P.tube(np.array([(0.11, 0.0, 0.44), (0.13, 0.0, 0.62), (0.10, 0.0, 0.76)]), 0.011, segments=10)
    wheels = []
    for x in (-0.16, 0.16):
        for y in (-0.10, 0.10):
            wheels.append(P.transform(P.lathe([(0.0, -0.015), (0.03, -0.015), (0.03, 0.015), (0.0, 0.015)], 14),
                                      R=Rx(math.pi / 2), t=(x, y, 0.03)))
    return body, water, P.merge(wringer, lever), P.merge(*wheels)


# ================================================================================================ decor
def pendant(drop, shade_r=0.17):
    """A pendant lamp hanging from the ceiling (origin at the ceiling): (metal: canopy, rod, dome shade outside),
    (shade inside), (bulb)."""
    canopy = P.lathe([(0.0, -0.035), (0.055, -0.035), (0.06, -0.025), (0.06, 0.0), (0.0, 0.0)], segments=24)
    rod = P.lathe([(0.006, -drop + 0.13), (0.006, -0.03)], segments=10)
    dome_out = [(shade_r, -drop), (shade_r - 0.01, -drop + 0.012), (0.13, -drop + 0.055), (0.08, -drop + 0.095),
                (0.030, -drop + 0.12), (0.012, -drop + 0.125)]
    dome_in = [(0.0, -drop + 0.112), (0.077, -drop + 0.094), (0.126, -drop + 0.056), (shade_r - 0.014, -drop + 0.013),
               (shade_r - 0.004, -drop + 0.001)]
    outside = P.lathe(dome_out, segments=36)
    inside = P.lathe(dome_in, segments=36)
    bulb = P.lathe([(0.0, -drop + 0.020), (0.030, -drop + 0.035), (0.034, -drop + 0.060), (0.018, -drop + 0.090),
                    (0.0, -drop + 0.095)], segments=16)
    return P.merge(canopy, rod, outside), inside, bulb


def planter(radius, h):
    """A ceramic planter (origin at its foot): (pot, soil)."""
    pot = P.lathe([(radius * 0.78, 0.0), (radius, h * 0.85), (radius * 1.04, h), (radius * 0.96, h),
                   (radius * 0.92, h * 0.9), (0.0, h * 0.9)], segments=40)
    soil = P.lathe([(0.0, h * 0.9 + 0.004), (radius * 0.93, h * 0.9 + 0.004)], segments=40)
    return pot, soil


def foliage(center, radius, z0, z1, seed, n=420, leaf=(0.075, 0.032)):
    """The leaves of a potted tree (fiddle-leaf fig / olive): `n` leaves, each a slightly folded lanceolate blade,
    scattered over an ellipsoidal crown; returns (leaves, trunk). Texture coordinates: u = which shade (0..1)."""
    rng = np.random.default_rng(seed)
    cx, cy = center
    cz, hz = (z0 + z1) / 2 + 0.15 * (z1 - z0), (z1 - z0) / 2 * 0.75
    V, F, N, UV = [], [], [], []
    for i in range(n):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        rr = rng.uniform(0.55, 1.0) ** 0.5
        p = np.array([cx + d[0] * radius * rr, cy + d[1] * radius * rr, cz + d[2] * hz * rr])
        L, W = leaf[0] * rng.uniform(0.75, 1.25), leaf[1] * rng.uniform(0.8, 1.2)
        out = np.array([d[0], d[1], 0.0])
        out = out / max(1e-9, np.linalg.norm(out))
        yaw = math.atan2(out[1], out[0]) + rng.normal(0, 0.6)
        droop = rng.uniform(-0.5, 0.9)
        R = Rz(yaw) @ Ry(droop)
        loc = np.array([[0.0, 0.0, 0.0], [L * 0.45, W / 2, 0.004], [L, 0.0, 0.0], [L * 0.45, -W / 2, 0.004],
                        [L * 0.45, 0.0, -0.003]])
        w = loc @ R.T + p
        base = len(V)
        shade = rng.random()
        for k, q in enumerate(w):
            V.append(q)
            UV.append((shade, 0.5))
        nrm = R @ np.array([0.0, 0.0, 1.0])
        if nrm[2] < 0:
            nrm = -nrm
        N += [nrm] * 5
        F += [(base, base + 1, base + 4), (base + 4, base + 1, base + 2), (base, base + 4, base + 3),
              (base + 4, base + 2, base + 3)]
    leaves = (np.array(V), np.array(F, np.int32), np.array(N), np.array(UV))
    # double-sided: the back faces too
    Vb, Fb, Nb, UVb = leaves
    leaves = P.merge(leaves, (Vb, Fb[:, ::-1].copy(), -Nb, UVb))
    trunk_pts = [(cx, cy, z0 - 0.02), (cx + 0.02, cy - 0.01, (z0 + cz) / 2), (cx - 0.01, cy + 0.02, cz)]
    trunk = P.tube(P.catmull_rom(trunk_pts, 5), 0.022, segments=10)
    return leaves, trunk


def frame_picture(w, h, depth=0.03, border=0.04):
    """A picture frame on a wall (facing +y, origin at its middle on the wall): (frame, canvas)."""
    parts = []
    for sgn in (-1.0, 1.0):
        parts.append(P.transform(P.box([w / 2, depth / 2, border / 2]), t=(0.0, depth / 2, sgn * (h / 2 - border / 2))))
        parts.append(P.transform(P.box([border / 2, depth / 2, h / 2 - border]), t=(sgn * (w / 2 - border / 2),
                                                                                    depth / 2, 0.0)))
    canvas = P.transform(P.quad(w - 2 * border, h - 2 * border, "+y"), t=(0.0, depth * 0.6, 0.0))
    return P.merge(*parts), canvas


def plate_stack(n, r=0.12, dz=0.016):
    parts = []
    for i in range(n):
        parts.append(P.transform(P.lathe([(0.0, 0.0), (r * 0.66, 0.0), (r, 0.014), (r * 0.96, 0.016),
                                          (0.0, 0.008)], segments=28), t=(0.0, 0.0, i * dz)))
    return P.merge(*parts)


def glass_rows(nx, ny, pitch, upside_down=True):
    """Rows of upturned tumblers on a shelf (one merged mesh)."""
    g = tumbler()
    if upside_down:
        g = P.transform(g, R=Rx(math.pi), t=(0.0, 0.0, 0.115))
    parts = []
    for i in range(nx):
        for j in range(ny):
            parts.append(P.transform(g, t=((i - (nx - 1) / 2) * pitch, (j - (ny - 1) / 2) * pitch, 0.0)))
    return P.merge(*parts)
