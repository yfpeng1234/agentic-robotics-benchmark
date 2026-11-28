"""One episode's lunch shift, drawn from its seed (the cell's scenario; the cell manual states the rules, never the
numbers of a drawn shift).

At the shift's start every table is in one of three states: dirty (its party has just left: their plates, glasses,
cutlery and napkins, crumbs and the soiling of the top are there), occupied (a party is finishing lunch: the same
things, the guests seated; the party leaves at a drawn time) or set (one roll-up at every seat). One floor spill
happens during the shift: a drink spilled in the aisle beside an occupied table, at a drawn time while that party is
still seated. The bus cart stands at its home by the service station (its tub empty, the cloth in the sanitizer
bucket, the crumb pan on its hook); sixteen clean roll-ups lie in the two baskets on the station's counter; the mop
hangs in its clip, the wet-floor cone stands on its spot; the robot starts beside the cart.
"""

from __future__ import annotations

import math

import numpy as np

from . import objects as O
from . import room as RM

PARK_AT = (3.0, 16.0, 0.5)        # where the bodies not in play wait (far north of the room, out of every view)
SOIL_CELL = 0.02
SPILL_CELL = 0.03
SHIFT_S = 4200.0                  # the shift (the episode's simulated time)
LEAVE_WINDOW = (240.0, 2300.0)    # when the occupied tables' parties leave
LEAVE_GAP = 200.0
SPILL_WINDOW = (300.0, 1800.0)


def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _qz(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def seat_index():
    return {st["id"]: (name, st) for name, t in RM.tables().items() for st in t["seats"]}


def _overlaps(xy, r, placed):
    return any(math.hypot(xy[0] - q[0], xy[1] - q[1]) < r + rq for q, rq in placed)


def draw(seed: int, nominal: bool = False) -> dict:
    rng = np.random.default_rng((int(seed) * 2654435761 + 12345) & ((1 << 64) - 1))
    tables = RM.tables()
    names = sorted(tables)
    if nominal:
        dirty, occupied = ["T2", "T5"], ["T1", "T4", "T7", "T8"]
    else:
        perm = [names[i] for i in rng.permutation(len(names))]
        n_dirty = int(rng.integers(1, 3))
        n_occ = int(rng.integers(3, 5))
        dirty, occupied = perm[:n_dirty], perm[n_dirty:n_dirty + n_occ]
    set_tables = [n for n in names if n not in dirty and n not in occupied]
    # departures of the occupied tables (sorted, at least LEAVE_GAP apart)
    for _ in range(200):
        t = np.sort(rng.uniform(*LEAVE_WINDOW, size=len(occupied)))
        if len(t) < 2 or np.min(np.diff(t)) >= LEAVE_GAP:
            break
    order = [occupied[i] for i in rng.permutation(len(occupied))]
    leave = {n: float(round(x, 1)) for n, x in zip(order, t)}
    state = {}
    for n in names:
        if n in dirty:
            state[n] = {"state": "dirty", "leave_s": 0.0}
        elif n in occupied:
            state[n] = {"state": "occupied", "leave_s": leave[n]}
        else:
            state[n] = {"state": "set", "leave_s": None}
    loose, objects, parked = {}, {}, []
    guests = {}
    crumbs, crumb_r, crumb_home = [], [], []
    soil_cells = []
    rollups = []
    # ---------------------------------------------------------------- the used tables (dirty and occupied)
    for n in dirty + occupied:
        t = tables[n]
        (cx, cy), (sx, sy) = t["center"], t["size"]
        zt = RM.TABLE_TOP_Z
        placed = []
        for st in t["seats"]:
            sid = st["id"]
            phi = st["facing"]
            Rg = _rot(phi)
            fwd, right = Rg @ np.array([1.0, 0.0]), Rg @ np.array([0.0, -1.0])
            setting = np.array(st["setting"]) + rng.uniform(-0.02, 0.02, size=2)
            outward = 1.0 if (st["chair"][0] - cx) >= 0 else -1.0     # four-tops: the glass towards the nearer end
            if t["kind"] == "four_top":
                glass_xy = setting + fwd * rng.uniform(0.06, 0.10) + np.array([outward * rng.uniform(0.15, 0.18), 0.0])
            else:
                glass_xy = setting + fwd * rng.uniform(0.04, 0.08) + right * rng.uniform(0.17, 0.20)
            food = O.FOODS[int(rng.integers(len(O.FOODS)))]
            loose[f"plate_{sid}"] = ("plate", {"food": food, "seed": int(rng.integers(1 << 30))})
            objects[f"plate_{sid}"] = ((float(setting[0]), float(setting[1]), zt + 0.002), _qz(rng.uniform(0, 6.28)))
            placed.append((tuple(setting), O.PLATE["r"]))
            drink = list(O.DRINKS)[int(rng.integers(len(O.DRINKS)))]
            loose[f"glass_{sid}"] = ("glass", {"drink": drink, "level": float(rng.uniform(0.025, 0.06))})
            objects[f"glass_{sid}"] = ((float(glass_xy[0]), float(glass_xy[1]), zt + 0.002), (1.0, 0.0, 0.0, 0.0))
            placed.append((tuple(glass_xy), O.GLASS["r"] + 0.01))
            # cutlery: on the plate in the finished position (both parallel), or put down on the table to the
            # plate's left (the fork outside the knife)
            on_plate = rng.random() < 0.8
            ang = phi + (-0.65 if on_plate else 0.0) + rng.normal(0, 0.12)
            ua, va = np.array([math.cos(ang), math.sin(ang)]), np.array([-math.sin(ang), math.cos(ang)])
            for k, kind in enumerate(("fork", "knife")):
                if on_plate:
                    xy = setting + va * (k * 0.025 - 0.012) + ua * 0.01
                    z = zt + 0.002 + O.PLATE["h"] + 0.001
                else:
                    xy = setting - right * (O.PLATE["r"] + 0.045 - 0.026 * k)
                    z = zt + 0.002
                    placed.append((tuple(xy), 0.02))
                yaw = ang
                loose[f"{kind}_{sid}"] = (kind, {})
                objects[f"{kind}_{sid}"] = ((float(xy[0]), float(xy[1]), z), _qz(yaw))
            # the napkin: crumpled beside the plate or dropped on it
            if rng.random() < 0.4:
                xy = setting + rng.uniform(-0.03, 0.03, size=2)
                z = zt + 0.002 + O.PLATE["h"] + (0.008 if on_plate else 0.0)
            else:
                for _ in range(40):
                    xy = setting + right * rng.uniform(-0.22, -0.16) + fwd * rng.uniform(-0.05, 0.08)
                    if abs(xy[0] - cx) < sx / 2 - 0.05 and abs(xy[1] - cy) < sy / 2 - 0.05 and \
                            not _overlaps(xy, 0.045, placed):
                        break
                z = zt + 0.002
                placed.append((tuple(xy), 0.045))
            loose[f"napkin_{sid}"] = ("napkin", {"seed": int(rng.integers(1 << 30)),
                                                 "colour": "sage" if rng.random() < 0.7 else "white"})
            objects[f"napkin_{sid}"] = ((float(xy[0]), float(xy[1]), z), _qz(rng.uniform(0, 6.28)))
            # the soiling: a smear or two by the plate, a ring under the glass
            for _ in range(int(rng.integers(1, 3))):
                c = setting + rng.uniform(-0.12, 0.12, size=2)
                a, b = rng.uniform(0.04, 0.08), rng.uniform(0.025, 0.05)
                th = rng.uniform(0, math.pi)
                soil_cells += _blob(n, c, a, b, th, rng, t)
            soil_cells += _ring(n, glass_xy, rng, t)
            if n in occupied:
                guests[sid] = {"skin": int(rng.integers(len(O.SKIN))), "hair": int(rng.integers(len(O.HAIR))),
                               "shirt": int(rng.integers(len(O.SHIRT))), "pants": int(rng.integers(len(O.TROUSERS))),
                               "scale": float(rng.uniform(0.93, 1.05)), "build": float(rng.uniform(0.9, 1.1)),
                               "lean": float(rng.uniform(-0.03, 0.05))}
        # crumbs over the top, more round the plates
        k_cr = int(rng.integers(12, 22))
        for _ in range(k_cr):
            for _ in range(30):
                if rng.random() < 0.7:
                    st = t["seats"][int(rng.integers(len(t["seats"])))]
                    xy = np.array(st["setting"]) + rng.normal(0, 0.09, size=2)
                else:
                    xy = np.array([cx, cy]) + rng.uniform(-1, 1, size=2) * (np.array([sx, sy]) / 2 - 0.04)
                if abs(xy[0] - cx) < sx / 2 - 0.02 and abs(xy[1] - cy) < sy / 2 - 0.02:
                    break
            r = float(rng.uniform(0.0035, 0.007))
            crumbs.append((float(xy[0]), float(xy[1]), zt + r * 0.5))
            crumb_r.append(r)
            crumb_home.append(n)
    # ---------------------------------------------------------------- the set tables: a roll-up at every seat
    k = 0
    for n in set_tables:
        for st in tables[n]["seats"]:
            x, y, yaw = st["rollup"]
            name = f"rollup_{k:02d}"
            rollups.append(name)
            loose[name] = ("rollup", {})
            yaw = yaw + rng.normal(0, 0.04)
            objects[name] = ((x + float(rng.normal(0, 0.006)), y + float(rng.normal(0, 0.006)), RM.TABLE_TOP_Z + 0.001),
                             _qz(yaw))
            k += 1
    # ---------------------------------------------------------------- the cart, its caddy and the tools
    cart = RM.CART["home"]
    cxw, cyw, cyaw = cart
    Rc = _rot(cyaw)

    def cart_pt(p):
        q = Rc @ np.array(p[:2]) + np.array([cxw, cyw])
        return (float(q[0]), float(q[1]), float(p[2]))
    C = RM.CART
    for bk in RM.BASKETS:
        (bxc, byc), (bsx, bsy), bh = bk["center"], bk["size"], bk["h"]
        z0b = RM.STATION["box"][5]
        for layer in range(2):
            for j in range(4):
                name = f"rollup_{k:02d}"
                rollups.append(name)
                loose[name] = ("rollup", {})
                xl = bxc - bsx / 2 + 0.006 + O.ROLLUP["r"] + j * 0.044 + layer * 0.022 * 0.0
                xl = bxc - 0.066 + j * 0.044
                objects[name] = ((xl, byc, z0b + 0.006 + 0.001 + layer * (2 * O.ROLLUP["r"] * 0.88 + 0.002)),
                                 _qz(math.pi / 2))
                k += 1
    bx, by = C["bucket_at"]
    loose["cloth"] = ("cloth", {})
    objects["cloth"] = (cart_pt((bx, by, C["top_z"] + RM.BUCKET["rack_h"] + 0.002 + O.CLOTH["size"][2] / 2)),
                        _qz(cyaw + 0.4))
    # the crumb pan hangs on its hook: lip down, the handle up
    hxp, hyp, hzp = C["pan_hook"]
    loose["crumb_pan"] = ("crumb_pan", {})
    objects["crumb_pan"] = pan_on_hook(cart)
    loose["cone"] = ("cone", {})
    objects["cone"] = ((RM.CONE_HOME[0], RM.CONE_HOME[1], 0.001), _qz(0.0))      # its bar along the wall
    loose["mop"] = ("mop", {})
    mx, my, mz = RM.MOP_HOME["clip"]
    objects["mop"] = ((mx - 0.03, my, 0.062), _qz(math.pi / 2))         # its head along the wall
    # parked bodies: lined up far north of the room
    for i, name in enumerate(parked):
        objects[name] = ((PARK_AT[0] + 0.3 * (i % 20), PARK_AT[1] + 0.3 * (i // 20), PARK_AT[2]), (1.0, 0.0, 0.0, 0.0))
    # ---------------------------------------------------------------- the spill
    spill = _spill(rng, state, tables, guests)
    # ---------------------------------------------------------------- the robot
    rx, ry, ryaw = RM.ROBOT_HOME
    if not nominal:
        rx += float(rng.uniform(-0.08, 0.08))
        ry += float(rng.uniform(-0.08, 0.08))
        ryaw += float(rng.uniform(-0.15, 0.15))
    return {"seed": int(seed), "nominal": bool(nominal), "tables": state, "loose": loose, "objects": objects,
            "parked": parked, "rollups": rollups, "guests": guests, "crumbs": crumbs, "crumb_r": crumb_r,
            "crumb_home": crumb_home, "soil": {"cell": SOIL_CELL, "cells": _dedup(soil_cells)}, "spill": spill,
            "cart": cart, "robot": (rx, ry, ryaw), "look_seed": int(rng.integers(1 << 30)), "shift_s": SHIFT_S}


def pan_on_hook(cart):
    """The crumb pan's pose hanging on the cart's hook below the push handle: its lip down, its floor against the
    cart's back end, its handle rising towards the pusher."""
    import mujoco
    cxw, cyw, cyaw = cart
    hxp, hyp, hzp = RM.CART["pan_hook"]
    base, axis = O.pan_handle_axis()
    grip_l = base + axis * 0.06                     # where the hook takes the handle (pan frame)
    Rc3 = np.array([[math.cos(cyaw), -math.sin(cyaw), 0], [math.sin(cyaw), math.cos(cyaw), 0], [0, 0, 1.0]])
    # the pan's axes in the cart frame: x (lip -> back wall) up, y along the cart's y, z outwards (the cart's -x)
    Rp = np.stack([np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0])], axis=1)
    hook = np.array([hxp, hyp, hzp])
    p_cart = hook - Rp @ grip_l
    pw = Rc3 @ p_cart + np.array([cxw, cyw, 0.0])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (Rc3 @ Rp).reshape(-1))
    return (tuple(float(v) for v in pw), tuple(float(v) for v in q))


def _cells_in(t, f):
    (cx, cy), (sx, sy) = t["center"], t["size"]
    c = SOIL_CELL
    out = []
    nx, ny = int(round(sx / c)), int(round(sy / c))
    for i in range(nx):
        for j in range(ny):
            x = cx - sx / 2 + (i + 0.5) * c
            y = cy - sy / 2 + (j + 0.5) * c
            g = f(x, y)
            if g > 0.05:
                out.append((x, y, float(min(1.0, g))))
    return out


def _blob(name, c, a, b, th, rng, t):
    ct, st_ = math.cos(th), math.sin(th)
    wob = rng.uniform(0, 2 * math.pi, size=3)

    def f(x, y):
        dx, dy = x - c[0], y - c[1]
        u, v = dx * ct + dy * st_, -dx * st_ + dy * ct
        ang = math.atan2(v, u)
        rr = 1.0 + 0.18 * math.sin(3 * ang + wob[0]) + 0.10 * math.sin(5 * ang + wob[1])
        q = (u / a) ** 2 + (v / b) ** 2
        return (1.0 - q / rr ** 2) * 1.6 if q < rr ** 2 else 0.0
    return [(name, x, y, g) for x, y, g in _cells_in(t, f)]


def _ring(name, c, rng, t):
    r0 = O.GLASS["r"] - 0.004

    def f(x, y):
        d = math.hypot(x - c[0], y - c[1])
        return 0.9 if abs(d - r0) < 0.011 else (0.35 if d < r0 else 0.0)
    return [(name, x, y, g) for x, y, g in _cells_in(t, f)]


def _dedup(cells):
    seen = {}
    for (n, x, y, g) in cells:
        key = (n, round(x, 4), round(y, 4))
        seen[key] = min(1.0, seen.get(key, 0.0) + g)
    return [(k[0], k[1], k[2], v) for k, v in sorted(seen.items())]


def aisle_spots(tables, name):
    """Candidate spill centres beside a table: in the aisle(s) next to its chairs, clear of the chairs and guests."""
    t = tables[name]
    (cx, cy), (sx, sy) = t["center"], t["size"]
    row = t["row"]
    out = []
    chair_reach = sy / 2 + RM.CHAIR["offset"] + 0.21 + 0.35      # the chairs' backs and the guests' room behind them
    if row in ("S", "M"):
        out.append((cx, cy + chair_reach + 0.25))                  # the aisle north of the table
    if row in ("M", "N"):
        out.append((cx, cy - chair_reach - 0.25))                  # the aisle south of the table
    return out


def _spill(rng, state, tables, guests):
    occ = [n for n, s in state.items() if s["state"] == "occupied"]
    for _ in range(100):
        n = occ[int(rng.integers(len(occ)))]
        t_sp = float(rng.uniform(*SPILL_WINDOW))
        if state[n]["leave_s"] is None or state[n]["leave_s"] < t_sp + 240.0:
            continue
        spots = aisle_spots(tables, n)
        c = np.array(spots[int(rng.integers(len(spots)))]) + np.array([rng.uniform(-0.35, 0.35),
                                                                        rng.uniform(-0.08, 0.08)])
        break
    else:
        n = occ[0]
        t_sp = SPILL_WINDOW[0]
        c = np.array(aisle_spots(tables, n)[0])
    R = float(rng.uniform(0.17, 0.26))
    lobes = [(c + rng.normal(0, R * 0.35, size=2), R * rng.uniform(0.45, 0.75)) for _ in range(4)]
    cells, g0 = [], []
    s = SPILL_CELL
    for i in range(-14, 15):
        for j in range(-14, 15):
            x, y = c[0] + i * s, c[1] + j * s
            inside = math.hypot(x - c[0], y - c[1]) < R * 0.8 or any(math.hypot(x - p[0], y - p[1]) < r
                                                                     for p, r in lobes)
            if inside:
                cells.append((float(x), float(y)))
                g0.append(float(rng.uniform(0.8, 1.0)))
    drink = ("cola", "orange_juice", "red_wine")[int(rng.integers(3))]
    return {"t": round(t_sp, 1), "table": n, "center": (float(c[0]), float(c[1])), "radius": R, "cell": s,
            "cells": cells, "g0": g0, "drink": drink, "volume_ml": float(round(len(cells) * s * s * 0.4 * 1e3, 1))}
