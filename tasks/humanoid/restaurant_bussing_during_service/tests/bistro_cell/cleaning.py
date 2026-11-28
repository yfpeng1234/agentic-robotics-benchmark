"""The shift's cleaning processes (the cell manual states the rules at a data sheet's level; the program is told none
of the states): the soiling of the tables, the sanitizer on the tops, the wiping cloth, the crumbs, the floor spill
and the mop.

Soil (food smears and drink rings on a table's top: a grid of 20 mm cells, soil per cell g in [0, 1]): the cloth
flat on the top (its face within 25 deg of it, pressed on it with a normal force above 0.5 N) and sliding takes soil
off every cell under its face: per tick g -> g * exp(-K_WIPE * min(1, N / N_FULL) * ds * f), ds the distance slid
(light pressure does it: from N_FULL = 4 N on, harder pressing adds nothing), f = (wet with sanitizer 1.0 | damp 0.35
| dry 0.12) x (1 - load / LOAD_MAX); what comes off loads the cloth, and a cloth near its load capacity smears soil
back where it passes (DEPOSIT x (load / LOAD_MAX)^3 per cloth length slid).
Sanitizer: the cloth dipped in the bucket's solution (its middle inside the bucket, below the solution's surface, for
DIP_S) is wet with sanitizer (charge 1) and sheds its soil; wiping uses the charge up (AREA_PER_CHARGE m^2 of top per
charge); a table's top counts as sanitized where the cloth passed pressed while it still held sanitizer (a grid of
50 mm cells over the top).
Crumbs (kinematic: drawn-only bodies the cell moves): they rest on the highest surface below them. The cloth pressed on
a surface pushes the crumbs under it ahead of it; a crumb pushed over the edge drops (a few centimetres on the way it
was pushed) onto what is below: the floor, a chair's seat, or the crumb pan if its opening is there (the pan within
35 deg of level); crumbs in the pan ride with it and pour out over its lowest edge when it is tipped past 55 deg.
A crumb that falls into the bus tub (or onto the cart's top shelf) rides with the cart; the dish crew empties the tub.
Spill (a drink on the floor: a grid of 30 mm cells, liquid per cell in [0, 1] of a 0.4 mm film): the mop's pad flat
on the floor (within 20 deg of level, its underside within 12 mm of it) and pressed on it (a normal force above
3 N) takes the liquid under it up as it moves: g -> g * exp(-K_MOP * ds) per tick, ds the distance moved, until the
pad holds PAD_CAPACITY; a cell whose liquid is below DRY_FRACTION of its start is damp and dries DRY_S later.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import objects as O
from . import room as RM

TICK = 0.01
K_WIPE = 20.0
N_MIN, N_FULL = 0.5, 4.0
FLAT_COS = math.cos(math.radians(25.0))
SANITIZER_F, DAMP_F, DRY_F = 1.0, 0.35, 0.12
LOAD_MAX = 120.0                   # 25 mm cells' worth of soil a cloth holds (a table and a half of smears)
DEPOSIT = 0.25
DIP_S = 1.0
AREA_PER_CHARGE = 1.6              # m^2 of top one dip wets
COVER_CELL = 0.05
EDGE_FLY = (0.01, 0.04)
SCATTER_SPEED = 1.0
PAN_LEVEL_COS = math.cos(math.radians(35.0))
PAN_SPILL_COS = math.cos(math.radians(55.0))
K_MOP = 9.0
MOP_N_MIN = 3.0
MOP_ZTOL = 0.012
MOP_LEVEL_COS = math.cos(math.radians(20.0))
PAD_CAPACITY = 900.0               # ml
FILM_MM = 0.4
DRY_FRACTION = 0.05
DRY_S = 90.0

FREE, PAN, CART, DELIVERED = 0, 1, 2, 3
STATE_NAMES = {FREE: "free", PAN: "pan", CART: "cart", DELIVERED: "delivered"}
CLOTH_RGB = np.array([0.86, 0.22, 0.20])
SOILED_RGB = np.array([0.42, 0.24, 0.14])


class Cloth:
    def __init__(self, m):
        self.m = m
        self.charge = 0.0          # sanitizer (1 = freshly dipped)
        self.wet = 1.0
        self.load = 0.0
        self.peak_load = 0.0
        self.dips = 0
        self.dip_s = 0.0
        self.area = 0.0
        self.geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cloth_vis")

    def sync(self):
        if self.geom < 0:
            return
        a = min(1.0, self.load / LOAD_MAX) ** 0.8
        self.m.geom_rgba[self.geom, :3] = CLOTH_RGB * (1 - a) + SOILED_RGB * a

    def record(self):
        return {"charge": round(self.charge, 3), "load": round(self.load, 3), "load_frac": round(self.load / LOAD_MAX, 3),
                "peak_load": round(self.peak_load, 3), "dips": self.dips, "area_m2": round(self.area, 3)}


class Soil:
    """The tables' soiling (20 mm cells) and the sanitizer coverage of every table's top (50 mm cells)."""

    def __init__(self, m, meta):
        sl = meta["soil"]
        self.m = m
        self.cell = float(sl["cell"])
        self.table = np.array([c[0] for c in sl["cells"]], dtype=object)
        self.xy = np.array([(c[1], c[2]) for c in sl["cells"]], float).reshape(-1, 2)
        self.g0 = np.array([c[3] for c in sl["cells"]], float)
        self.g = self.g0.copy()
        self.geom0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "soil0") if len(self.g) else -1
        self._shown = np.full(len(self.g), -1.0)
        self.unit = (self.cell / 0.025) ** 2
        self.removed = 0.0
        self.deposited = 0.0
        # sanitizer coverage
        self.cover = {}
        for name, t in RM.tables().items():
            (cx, cy), (sx, sy) = t["center"], t["size"]
            nx, ny = int(round(sx / COVER_CELL)), int(round(sy / COVER_CELL))
            xs = cx - sx / 2 + (np.arange(nx) + 0.5) * sx / nx
            ys = cy - sy / 2 + (np.arange(ny) + 0.5) * sy / ny
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            self.cover[name] = {"xy": np.stack([X.ravel(), Y.ravel()], 1), "done": np.zeros(nx * ny, bool),
                                "t_last": np.full(nx * ny, -1.0)}

    def residual(self, table=None):
        sel = self.table == table if table is not None else slice(None)
        return float(self.g[sel].sum()) * self.unit

    def total0(self, table=None):
        sel = self.table == table if table is not None else slice(None)
        return float(self.g0[sel].sum()) * self.unit

    def removed_fraction(self, table):
        t0 = self.total0(table)
        return 1.0 - self.residual(table) / t0 if t0 > 1e-9 else 1.0

    def coverage(self, table):
        return float(self.cover[table]["done"].mean())

    @staticmethod
    def _under(xy, c, R, half):
        d = xy - c[:2][None, :]
        a1 = R[:2, 0] / max(1e-9, np.linalg.norm(R[:2, 0]))
        a2 = np.array([-a1[1], a1[0]])
        return (np.abs(d @ a1) <= half[0]) & (np.abs(d @ a2) <= half[1])

    def wipe(self, table, cloth: Cloth, c, R, half, N, ds, t):
        if ds <= 0:
            return 0.0
        # sanitizer coverage
        cv = self.cover[table]
        sel_c = self._under(cv["xy"], c, R, half)
        if cloth.charge > 0.05 and sel_c.any():
            new = sel_c & ~cv["done"]
            cv["done"] |= sel_c
            cv["t_last"][sel_c] = t
            _ = new
        cloth.charge = max(0.0, cloth.charge - (2 * half[1] * ds) / AREA_PER_CHARGE)
        cloth.area += 2 * half[1] * ds
        # soil
        if not len(self.g):
            return 0.0
        sel = (self.table == table) & self._under(self.xy, c, R, half)
        if not sel.any():
            return 0.0
        f = SANITIZER_F if cloth.charge > 0.05 else (DAMP_F if cloth.wet > 0.5 else DRY_F)
        cap = max(0.0, 1.0 - cloth.load / LOAD_MAX)
        k = K_WIPE * min(1.0, N / N_FULL) * ds * f * cap
        g = self.g[sel]
        taken = g * (1.0 - math.exp(-k))
        g = g - taken
        lf = cloth.load / LOAD_MAX
        dep = DEPOSIT * lf ** 3 * (ds / O.CLOTH["size"][0])
        put = np.minimum(np.maximum(0.0, dep * (1.0 - g)), 1.0 - g)
        self.g[sel] = g + put
        t_sum, p_sum = float(taken.sum()) * self.unit, float(put.sum()) * self.unit
        cloth.load = max(0.0, cloth.load + t_sum - p_sum)
        cloth.peak_load = max(cloth.peak_load, cloth.load)
        self.removed += t_sum
        self.deposited += p_sum
        return t_sum

    def sync(self, force=False):
        if not len(self.g):
            return
        a = 0.85 * np.power(np.clip(self.g, 0.0, 1.0), 0.6)
        a[self.g < 0.01] = 0.0
        ch = np.abs(a - self._shown) > 0.01 if not force else np.ones(len(a), bool)
        idx = np.nonzero(ch)[0]
        if len(idx):
            self.m.geom_rgba[self.geom0 + idx, 3] = a[idx]
            self._shown[idx] = a[idx]


class Spill:
    """The floor spill: hidden until it happens, then a film the mop takes up; damp cells dry DRY_S later."""

    def __init__(self, m, meta):
        sp = meta["spill"]
        self.m = m
        self.t_spill = float(sp["t"])
        self.center = np.array(sp["center"], float)
        self.cell = float(sp["cell"])
        self.xy = np.array(sp["cells"], float).reshape(-1, 2)
        self.g0 = np.array(sp["g0"], float)
        self.g = np.zeros(len(self.g0))
        self.active = False
        self.damp_since = np.full(len(self.g0), -1.0)
        self.geom0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "spill0")
        self.rgb = None
        self._shown = np.full(len(self.g0), -1.0)
        self.ml_per_cell = self.cell * self.cell * FILM_MM * 1e3        # ml of a full cell (m^2 x mm = litres / 1e3)
        self.pad = 0.0                                                   # ml in the mop's pad
        self.mopped_ml = 0.0
        self.mop_s = 0.0
        self.dry_at = None

    def start(self, t):
        self.active = True
        self.g = self.g0.copy()

    def volume_ml(self):
        return float(self.g.sum()) * self.ml_per_cell

    def removed_fraction(self):
        t0 = float(self.g0.sum())
        return 1.0 - float(self.g.sum()) / t0 if (self.active and t0 > 0) else 0.0

    def wet_cells(self):
        return self.g > DRY_FRACTION * np.maximum(self.g0, 1e-9)

    def is_dry(self, t):
        if not self.active:
            return False
        if self.wet_cells().any():
            return False
        return bool(np.all(self.damp_since >= 0) and np.all(t - self.damp_since >= DRY_S))

    def mop(self, c, R, half, ds):
        if not self.active or ds <= 0:
            return 0.0
        d = self.xy - c[:2][None, :]
        a1 = R[:2, 0] / max(1e-9, np.linalg.norm(R[:2, 0]))
        a2 = np.array([-a1[1], a1[0]])
        sel = (np.abs(d @ a1) <= half[0]) & (np.abs(d @ a2) <= half[1])
        if not sel.any():
            return 0.0
        room = max(0.0, 1.0 - self.pad / PAD_CAPACITY)
        g = self.g[sel]
        taken = g * (1.0 - math.exp(-K_MOP * ds * room))
        self.g[sel] = g - taken
        ml = float(taken.sum()) * self.ml_per_cell
        self.pad += ml
        self.mopped_ml += ml
        return ml

    def tick(self, t):
        if not self.active:
            return
        wet = self.wet_cells()
        newly = (~wet) & (self.damp_since < 0)
        self.damp_since[newly] = t
        if self.dry_at is None and self.is_dry(t):
            self.dry_at = t

    def sync(self, t):
        if not len(self.g):
            return
        if not self.active:
            a = np.zeros(len(self.g))
        else:
            a = 0.88 * np.power(np.clip(self.g / np.maximum(self.g0, 1e-9), 0.0, 1.0), 0.5)
            damp = (~self.wet_cells()) & (self.damp_since >= 0) & (t - self.damp_since < DRY_S)
            a[damp] = np.maximum(a[damp], 0.18 * (1.0 - (t - self.damp_since[damp]) / DRY_S))
        ch = np.abs(a - self._shown) > 0.01
        idx = np.nonzero(ch)[0]
        if len(idx):
            self.m.geom_rgba[self.geom0 + idx, 3] = a[idx]
            self._shown[idx] = a[idx]


class Crumbs:
    def __init__(self, m, d, meta, layout, rng):
        self.m, self.d = m, d
        n = meta["crumbs"]
        self.n = n
        self.P = np.array(layout["crumbs"], float).reshape(n, 3)
        self.r = np.array(layout["crumb_r"], float)
        self.home = np.array(layout["crumb_home"], dtype=object)
        self.state = np.zeros(n, int)
        self.local = np.zeros((n, 3))
        self.mocap = np.array([int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"crumb{i}")])
                               for i in range(n)], int)
        self.on = np.array([f"{h}_top" for h in self.home], dtype=object)
        self.rng = rng
        self.surfaces = RM.surfaces()
        self.events = {"to_pan": 0, "to_cart": 0, "to_floor": 0, "to_chair": 0, "spilled": 0, "scattered": 0,
                       "delivered": 0}
        B = mujoco.mjtObj.mjOBJ_BODY
        self.pan_body = mujoco.mj_name2id(m, B, "crumb_pan")
        self.cart_body = mujoco.mj_name2id(m, B, "cart")
        self.cart_rec = meta["cart"]
        self.sync()

    def cart_frame(self):
        b = self.cart_body
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def pan_frame(self):
        b = self.pan_body
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def _in_pan(self, x, y, z, p, R):
        if R[2, 2] < PAN_LEVEL_COS or z < p[2]:
            return None
        d = np.array([x - p[0], y - p[1], 0.0])
        lx = float(d @ R[:, 0]) / max(1e-6, math.hypot(R[0, 0], R[1, 0]))
        ly = float(d @ R[:, 1]) / max(1e-6, math.hypot(R[0, 1], R[1, 1]))
        W, D = O.PAN["width"], O.PAN["depth"]
        if 0.004 < lx < D - 0.006 and abs(ly) < W / 2 - 0.006:
            return lx, ly
        return None

    def drop(self, i, x, y, z):
        """Crumb i falls from (x, y, z) onto the highest thing below it: the crumb pan, the cart's tub or top shelf, a
        table, a chair's seat, the station, the pass, the floor."""
        r = self.r[i]
        p, R = self.pan_frame()
        hit = self._in_pan(x, y, z, p, R)
        if hit is not None:
            self.state[i] = PAN
            self.local[i] = (hit[0], hit[1], O.PAN["t"] + r * 0.55)
            self.events["to_pan"] += 1
            return
        pc, Rc = self.cart_frame()
        loc = Rc.T @ (np.array([x, y, z]) - pc)
        sx, sy = RM.CART["size"]
        if abs(loc[0]) < sx / 2 and abs(loc[1]) < sy / 2 and z > RM.CART["top_z"] - 0.01:
            x0, y0, z0, x1, y1, z1 = self.cart_rec["tub_inside"]
            if x0 < loc[0] < x1 and y0 < loc[1] < y1:
                loc[2] = z0 + r * 0.55
            else:
                loc[2] = RM.CART["top_z"] + r * 0.55
            self.state[i] = CART
            self.local[i] = loc
            self.events["to_cart"] += 1
            return
        best, on = 0.0, "floor"
        for name, ((x0, y0, x1, y1), h) in self.surfaces.items():
            if x0 < x < x1 and y0 < y < y1 and h <= z + 1e-4 and h > best:
                best, on = h, name
        if on == "floor":
            self.events["to_floor"] += 1
        elif on.startswith("chair"):
            self.events["to_chair"] += 1
        self.state[i] = FREE
        self.on[i] = on
        self.P[i] = (x, y, best + r * 0.55)

    def plow(self, surface, c, R, half, v):
        idx = np.nonzero((self.state == FREE) & (self.on == surface))[0]
        if not len(idx):
            return
        sp = float(math.hypot(v[0], v[1]))
        if sp < 0.004:
            return
        a1 = R[:2, 0] / max(1e-9, np.linalg.norm(R[:2, 0]))
        a2 = np.array([-a1[1], a1[0]])
        d = self.P[idx, :2] - c[None, :2]
        u, w = d @ a1, d @ a2
        hu, hw = half
        inside = (np.abs(u) <= hu + self.r[idx]) & (np.abs(w) <= hw + self.r[idx])
        if not inside.any():
            return
        vu, vw = float(v[:2] @ a1), float(v[:2] @ a2)
        for k in np.nonzero(inside)[0]:
            i = idx[k]
            ri = self.r[i]
            if sp > SCATTER_SPEED:
                ang = math.atan2(v[1], v[0]) + self.rng.uniform(-0.6, 0.6)
                dist = self.rng.uniform(0.10, 0.40)
                nx, ny = self.P[i, 0] + dist * math.cos(ang), self.P[i, 1] + dist * math.sin(ang)
                self.events["scattered"] += 1
            else:
                uu, ww = u[k], w[k]
                if abs(vw) >= 0.25 * abs(vu):
                    ww = math.copysign(hw + ri + 0.003, vw)
                    uu = uu + self.rng.normal(0, 0.003)
                else:
                    uu = math.copysign(hu + ri + 0.003, vu)
                nx, ny = (c[0] + uu * a1[0] + ww * a2[0], c[1] + uu * a1[1] + ww * a2[1])
            self._move(i, nx, ny, surface)

    def _move(self, i, x, y, surface):
        (x0, y0, x1, y1), h = self.surfaces[surface]
        if x0 < x < x1 and y0 < y < y1:
            self.P[i, :2] = (x, y)
            return
        dx, dy = x - float(self.P[i, 0]), y - float(self.P[i, 1])
        n = math.hypot(dx, dy)
        if n > 1e-6:
            fly = float(self.rng.uniform(*EDGE_FLY))
            x, y = x + dx / n * fly, y + dy / n * fly
        self.drop(i, x, y, h)

    def carry(self):
        """Crumbs in the pan or on the cart ride with them; the pan tipped past 55 deg pours them out."""
        idx = np.nonzero(self.state == PAN)[0]
        if len(idx):
            p, R = self.pan_frame()
            if R[2, 2] < PAN_SPILL_COS:
                W, D, H = O.PAN["width"], O.PAN["depth"], O.PAN["wall"]
                edges = {"lip": np.array([0.0, 0.0, 0.0]), "back": np.array([D, 0.0, H]),
                         "left": np.array([D / 2, W / 2, H * 0.4]), "right": np.array([D / 2, -W / 2, H * 0.4])}
                low = min(edges, key=lambda k: float((p + R @ edges[k])[2]))
                for i in idx:
                    loc = self.local[i].copy()
                    if low == "lip":
                        loc[0] = -0.01
                    elif low == "back":
                        loc[0] = D + 0.01
                    else:
                        loc[1] = math.copysign(W / 2 + 0.01, edges[low][1])
                    wp = p + R @ loc
                    self.events["spilled"] += 1
                    self.state[i] = FREE
                    self.drop(i, float(wp[0]), float(wp[1]), float(wp[2]))
                idx = np.nonzero(self.state == PAN)[0]
            if len(idx):
                self.P[idx] = p[None, :] + self.local[idx] @ R.T
        idx = np.nonzero(self.state == CART)[0]
        if len(idx):
            pc, Rc = self.cart_frame()
            self.P[idx] = pc[None, :] + self.local[idx] @ Rc.T

    def deliver_tub(self, park_xyz):
        """The dish crew empties the tub: the crumbs in it go with the dishes."""
        x0, y0, z0, x1, y1, z1 = self.cart_rec["tub_inside"]
        k = 0
        for i in np.nonzero(self.state == CART)[0]:
            lx, ly = self.local[i, 0], self.local[i, 1]
            if x0 < lx < x1 and y0 < ly < y1:
                self.state[i] = DELIVERED
                self.P[i] = (park_xyz[0] + 0.01 * (i % 50), park_xyz[1] + 2.0, park_xyz[2])
                k += 1
        self.events["delivered"] += k
        return k

    def sync(self):
        self.d.mocap_pos[self.mocap] = self.P

    def counts(self):
        out = {STATE_NAMES[k]: int((self.state == k).sum()) for k in STATE_NAMES}
        free = self.state == FREE
        out["on_floor"] = int((free & (self.on == "floor")).sum())
        out["on_chairs"] = int((free & np.array([str(s).startswith("chair") for s in self.on])).sum())
        out["on_tables"] = int((free & np.array([str(s).endswith("_top") for s in self.on])).sum())
        return out


class Process:
    """Per tick, after the physics: wiping, dipping, the crumbs, mopping; then the drawn state."""

    def __init__(self, m, d, meta, layout, grips, rng):
        self.m, self.d = m, d
        self.grips = grips
        self.cloth = Cloth(m)
        self.soil = Soil(m, meta)
        self.spill = Spill(m, meta)
        self.crumbs = Crumbs(m, d, meta, layout, rng)
        G, B = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
        self.cloth_body = mujoco.mj_name2id(m, B, "cloth")
        self.cloth_geom = mujoco.mj_name2id(m, G, "cloth_col")
        self.surface_geoms = {}
        for name in RM.tables():
            self.surface_geoms[mujoco.mj_name2id(m, G, f"{name}_top_col")] = (f"{name}_top", name)
        for sid in [st["id"] for t in RM.tables().values() for st in t["seats"]]:
            g = mujoco.mj_name2id(m, G, f"chair_{sid}_seat_col")
            if g >= 0:
                self.surface_geoms[g] = (f"chair_{sid}", None)
        self.pad_geom = mujoco.mj_name2id(m, G, "mop_pad_col")
        self.head_body = mujoco.mj_name2id(m, B, "mop_head")
        self.floor_geom = mujoco.mj_name2id(m, G, "floor")
        self.bucket = meta["cart"]["bucket"]
        self.cart_body = mujoco.mj_name2id(m, B, "cart")
        self._prev_cloth = None
        self._prev_pad = None
        self.wipe_s = {}
        self.soil.sync(force=True)
        self.cloth.sync()

    def _contact_force(self, geom, others):
        """(surface key, total normal force) of `geom`'s contacts with any geom in `others` (a dict geom -> key)."""
        m, d = self.m, self.d
        f6 = np.zeros(6)
        tot = {}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == geom and g2 in others:
                o = g2
            elif g2 == geom and g1 in others:
                o = g1
            else:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            k = others[o]
            tot[k] = tot.get(k, 0.0) + abs(float(f6[0]))
        if not tot:
            return None, 0.0
        k = max(tot, key=tot.get)
        return k, tot[k]

    def in_bucket(self, c):
        bx, by, z0, r, fill = self.bucket
        pc = self.d.xpos[self.cart_body]
        Rc = self.d.xmat[self.cart_body].reshape(3, 3)
        loc = Rc.T @ (c - pc)
        return math.hypot(loc[0] - bx, loc[1] - by) < r - 0.02 and loc[2] < z0 + fill

    def tick(self, t):
        d = self.d
        cl = self.cloth
        # ---- the cloth on a surface
        b = self.cloth_body
        c, R = d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()
        if self._prev_cloth is None:
            self._prev_cloth = c.copy()
        v = (c - self._prev_cloth) / TICK
        ds = float(math.hypot(v[0], v[1])) * TICK
        self._prev_cloth = c.copy()
        key, N = self._contact_force(self.cloth_geom, self.surface_geoms)
        if key is not None and abs(R[2, 2]) > FLAT_COS:
            surf, table = key
            lx, ly, _ = O.CLOTH["size"]
            half = (lx / 2, ly / 2)
            if table is not None and N > N_MIN:
                self.soil.wipe(table, cl, c, R, half, N, ds, t)
                self.wipe_s[table] = self.wipe_s.get(table, 0.0) + TICK
            if N > 0.5:
                self.crumbs.plow(surf, c, R, half, v)
        # ---- the cloth in the sanitizer bucket
        if self.in_bucket(c):
            cl.dip_s += TICK
            if cl.dip_s >= DIP_S:
                if cl.charge < 0.99:
                    cl.dips += 1
                cl.charge = 1.0
                cl.wet = 1.0
                cl.load *= math.exp(-TICK / 0.8)
        else:
            cl.dip_s = 0.0
        # ---- the mop's pad on the floor
        if self.pad_geom >= 0 and self.spill.active:
            hb = self.head_body
            Rh = d.xmat[hb].reshape(3, 3)
            bottom = d.xpos[hb] + Rh @ np.array([0.0, 0.0, -0.046])
            if self._prev_pad is None:
                self._prev_pad = bottom.copy()
            vb = (bottom - self._prev_pad) / TICK
            self._prev_pad = bottom.copy()
            if bottom[2] < MOP_ZTOL and Rh[2, 2] > MOP_LEVEL_COS:
                _, Nf = self._contact_force(self.pad_geom, {self.floor_geom: "floor"})
                if Nf > MOP_N_MIN:
                    px, py, _ = O.MOP["pad"]
                    self.spill.mop(bottom, Rh, (px / 2, py / 2), float(math.hypot(vb[0], vb[1])) * TICK)
                    self.spill.mop_s += TICK
        self.spill.tick(t)
        # ---- crumbs ride with the pan and the cart
        self.crumbs.carry()
        # ---- what the cameras see
        self.soil.sync()
        self.spill.sync(t)
        cl.sync()
        self.crumbs.sync()

    def record(self):
        return {"soil_total0": round(self.soil.total0(), 3), "soil_residual": round(self.soil.residual(), 3),
                "soil_taken": round(self.soil.removed, 3), "soil_smeared": round(self.soil.deposited, 3),
                "cloth": self.cloth.record(), "crumbs": self.crumbs.counts(), "crumb_events": dict(self.crumbs.events),
                "wipe_s": {k: round(v, 2) for k, v in self.wipe_s.items()},
                "spill": {"active": self.spill.active, "volume_ml": round(self.spill.volume_ml(), 1),
                          "removed_fraction": round(self.spill.removed_fraction(), 4),
                          "mopped_ml": round(self.spill.mopped_ml, 1), "pad_ml": round(self.spill.pad, 1),
                          "mop_s": round(self.spill.mop_s, 2), "dry_at": self.spill.dry_at}}
