"""The shift's service score (grader only): the floor manager's checklist, read from the cell's state when the episode
ends (and from what the cell's monitors recorded during the shift). Credit out of TOTAL = 10:
  * tables turned, 6.0, shared equally by the tables whose party left (at the start or during the shift); per table
    (its share x):
      - cleared 0.45: the share of its guests' things (each seat's plate, glass, fork, knife and napkin) that are off
        the table and in the bus tub or with the dish crew;
      - wiped 0.25: half for the soil (food smears, drink rings) taken off, min(1, removed / 0.9), half for the top
        sanitized, min(1, covered / 0.9) (the share of the top the cloth passed over pressed while it held
        sanitizer, after the party left);
      - crumbs 0.05: the share of the table's crumbs caught in the crumb pan or the tub (or gone with the crew);
      - reset 0.25: once the table is cleared, the share of its seats with a roll-up lying in place: on the top in
        front of the seat, its middle 5-14 cm in from the edge and within 7 cm of the seat's middle, its axis within
        20 deg of square to the edge;
  * the spill, 2.0: the wet-floor cone standing beside it (upright, 0.3-1.2 m from the spill's middle, off the wet
    floor) within 120 s of the spill 0.5 (within 300 s 0.25); the spill mopped up, 1.0 x min(1, removed / 0.95);
    the cone back on its spot at the end 0.25, if it stayed beside the spill until the floor was dry; the mop back
    in its clip 0.25, once it was used;
  * dish return, 1.0: the share of the cleared tables' things the dish crew took;
  * closing, 1.0, for what was used during the shift: the cart at its home 0.4 (within 0.3 m, its heading within
    15 deg; once it was pushed 2 m), the cloth in the sanitizer bucket 0.3 (once it wiped), the crumb pan on its hook
    0.3 (once it was taken off).
Charged (PENALTY): each broken plate or glass 0.25 (besides its lost credit), each thing left on the floor 0.25 (the
guests' things, the roll-ups, the cloth, the crumb pan), each of a seated party's things a hand took 0.5; the
episode's credit after the charges is at least 0.
A safety violation zeroes the episode (the grader applies it).
"""

from __future__ import annotations

import math

import numpy as np

from . import cleaning as CL
from . import objects as O
from . import room as RM

TABLE_SHARE = 6.0
TABLE_PARTS = {"cleared": 0.45, "wiped": 0.25, "crumbs": 0.05, "reset": 0.25}
SPILL_PARTS = {"cone": 0.5, "mopped": 1.0, "cone_back": 0.25, "mop_back": 0.25}
RETURN_W = 1.0
CLOSE_PARTS = {"cart_home": 0.4, "cloth": 0.3, "pan": 0.3}
TOTAL = TABLE_SHARE + sum(SPILL_PARTS.values()) + RETURN_W + sum(CLOSE_PARTS.values())
# the checklist's parts as the record's credit_by_part names them, and their weights (out of TOTAL = 10)
WEIGHTS = {"tables": TABLE_SHARE, "spill": float(sum(SPILL_PARTS.values())), "dish_return": RETURN_W,
           "closing": float(sum(CLOSE_PARTS.values()))}
SOIL_FULL, COVER_FULL, SPILL_FULL = 0.90, 0.90, 0.95
ROLLUP_LAT, ROLLUP_ANG = 0.07, math.radians(20.0)
CONE_NEAR = (0.30, 1.20)
CONE_IN_TIME = (120.0, 300.0)
CONE_UPRIGHT_COS = math.cos(math.radians(15.0))
PENALTY = {"broken": 0.25, "on_floor": 0.25, "rude": 0.5}
CART_USED_M, CLOTH_USED_M2 = 2.0, 0.05       # closing counts what was used: the cart pushed 2 m, the cloth wiped 0.05 m^2
FLOOR_Z = 0.15


def guest_items(cell, table):
    ids = {st["id"] for st in RM.tables()[table]["seats"]}
    return sorted(n for n, (k, _p) in cell.layout["loose"].items() if k in O.GUEST_KINDS and n.split("_", 1)[1] in ids)


def turned_tables(cell):
    """The tables to turn in this shift: dirty at the start, or whose party leaves before the shift's end (by the
    schedule: a program that stops early does not take the later tables out of the count)."""
    cap = float(cell.sim_cap)
    return sorted(n for n, s in cell.layout["tables"].items() if s["state"] == "dirty"
                  or (s["state"] == "occupied" and s.get("leave_s") is not None and float(s["leave_s"]) < cap))


def where(cell, name):
    """'delivered' (taken by the crew), 'tub', 'floor', 'table:<T>', 'held' or 'other'."""
    if name in cell.shift.parked:
        return "delivered"
    if name in cell.grips.held():
        return "held"
    p, _ = cell.pose(name)
    loc = cell.cart._to_cart(p)
    x0, y0, z0, x1, y1, z1 = cell.meta["cart"]["tub_inside"]
    if x0 - 0.01 < loc[0] < x1 + 0.01 and y0 - 0.01 < loc[1] < y1 + 0.01 and z0 - 0.03 < loc[2] < z1 + 0.12:
        return "tub"
    if p[2] < FLOOR_Z:
        return "floor"
    for name_t, t in RM.tables().items():
        (cx, cy), (sx, sy) = t["center"], t["size"]
        if abs(p[0] - cx) < sx / 2 + 0.02 and abs(p[1] - cy) < sy / 2 + 0.02 and abs(p[2] - RM.TABLE_TOP_Z) < 0.12:
            return f"table:{name_t}"
    return "other"


def rollup_ok(cell, name, seat):
    """A roll-up lying in place for `seat` (see the docstring)."""
    p, R = cell.pose(name)
    t = None
    for tn, tt in RM.tables().items():
        if any(st["id"] == seat["id"] for st in tt["seats"]):
            t = tt
    (cx, cy), (sx, sy) = t["center"], t["size"]
    if abs(p[2] - RM.TABLE_TOP_Z) > 0.03 or R[2, 2] < 0.9:
        return False
    sgn = 1.0 if seat["side"] == "N" else -1.0
    edge = cy + sgn * sy / 2
    depth = sgn * (edge - p[1])                     # how far in from the seat's edge
    if not (RM.ROLLUP_FROM_EDGE[0] <= depth <= RM.ROLLUP_FROM_EDGE[1]):
        return False
    if abs(p[0] - seat["chair"][0]) > ROLLUP_LAT:
        return False
    ax = R[:2, 0]
    ang = math.atan2(abs(ax[0]), abs(ax[1]))         # 0 when the axis is along y (square to the edge)
    return ang <= ROLLUP_ANG


def score_cell(cell) -> dict:
    credit = {}
    detail = {"tables": {}}
    pr = cell.process
    # ------------------------------------------------------------------------ tables
    turned = turned_tables(cell)
    share = TABLE_SHARE / max(1, len(turned))
    used_rollups = set()
    tsum = 0.0
    for T in turned:
        items = guest_items(cell, T)
        loc = {n: where(cell, n) for n in items}
        off = [n for n, w in loc.items() if w in ("tub", "delivered")]
        cleared = len(off) / max(1, len(items))
        soil = pr.soil.removed_fraction(T)
        cover = pr.soil.coverage(T)
        wiped = 0.5 * min(1.0, soil / SOIL_FULL) + 0.5 * min(1.0, cover / COVER_FULL)
        st = pr.crumbs.state
        mine = pr.crumbs.home == T
        caught = int(((st == CL.PAN) | (st == CL.CART) | (st == CL.DELIVERED))[mine].sum())
        crumbs = caught / max(1, int(mine.sum()))
        seats = RM.tables()[T]["seats"]
        placed = {}
        if cleared >= 1.0:
            for seat in seats:
                for n in cell.layout["rollups"]:
                    if n in used_rollups or n in cell.shift.parked:
                        continue
                    if rollup_ok(cell, n, seat):
                        placed[seat["id"]] = n
                        used_rollups.add(n)
                        break
        reset = len(placed) / max(1, len(seats))
        parts = {"cleared": cleared, "wiped": wiped, "crumbs": crumbs, "reset": reset}
        c = share * sum(TABLE_PARTS[k] * v for k, v in parts.items())
        tsum += c
        detail["tables"][T] = {"left_at": cell.shift.left_at.get(T), "parts": {k: round(v, 4) for k, v in parts.items()},
                               "credit": round(c, 4), "soil_removed": round(soil, 4), "covered": round(cover, 4),
                               "rollups": placed, "items": loc}
    credit["tables"] = tsum
    # ------------------------------------------------------------------------ the spill
    mon = cell.shift.monitor
    sp = pr.spill
    spill = {}
    t_sp = sp.t_spill
    if sp.active:
        tc = mon.get("cone_placed_at")
        if tc is not None and tc - t_sp <= CONE_IN_TIME[0]:
            spill["cone"] = SPILL_PARTS["cone"]
        elif tc is not None and tc - t_sp <= CONE_IN_TIME[1]:
            spill["cone"] = SPILL_PARTS["cone"] / 2
        else:
            spill["cone"] = 0.0
        removed = sp.removed_fraction()
        spill["mopped"] = SPILL_PARTS["mopped"] * min(1.0, removed / SPILL_FULL)
        cone_home = cell.shift.cone_home()
        stayed = tc is not None and (sp.dry_at is not None and (mon.get("cone_left_at") or 1e18) >= sp.dry_at)
        spill["cone_back"] = SPILL_PARTS["cone_back"] if (cone_home and stayed) else 0.0
        spill["mop_back"] = SPILL_PARTS["mop_back"] if (cell.clips.engaged.get("mop") and sp.mop_s > 1.0) else 0.0
    else:
        removed = 0.0
        spill = {k: 0.0 for k in SPILL_PARTS}
    credit["spill"] = float(sum(spill.values()))
    # ------------------------------------------------------------------------ dish return
    all_items = [n for T in turned for n in guest_items(cell, T)]
    delivered = [n for n in all_items if n in cell.shift.parked]
    credit["dish_return"] = RETURN_W * len(delivered) / max(1, len(all_items))
    # ------------------------------------------------------------------------ closing
    close = {}
    x, y, yaw = cell.cart.pose()
    hx, hy, hyaw = RM.CART["home"]
    # (each for what was used during the shift: the cart pushed, the cloth wiping, the pan taken off its hook)
    cart_used = float(cell.cart.travel) > CART_USED_M
    close["cart_home"] = CLOSE_PARTS["cart_home"] if (cart_used and math.hypot(x - hx, y - hy) < 0.30 and
                                                      abs((yaw - hyaw + math.pi) % (2 * math.pi) - math.pi)
                                                      < math.radians(15.0) and not cell.cart.grasp) else 0.0
    cloth_used = float(pr.cloth.area) > CLOTH_USED_M2
    cloth_in = pr.in_bucket(cell.d.xpos[pr.cloth_body]) and "cloth" not in cell.grips.held()
    close["cloth"] = CLOSE_PARTS["cloth"] if (cloth_used and cloth_in) else 0.0
    pan_used = any(len(e) >= 3 and e[1] == "out" and e[2] == "crumb_pan" for e in cell.clips.events)
    close["pan"] = CLOSE_PARTS["pan"] if (pan_used and cell.clips.engaged.get("crumb_pan")) else 0.0
    credit["closing"] = float(sum(close.values()))
    # ------------------------------------------------------------------------ charges
    broken = sorted(n for n in cell.fragile.broken)
    on_floor = []
    for name, (kind, _p) in cell.layout["loose"].items():
        if kind in ("cone", "mop") or name in cell.shift.parked:
            continue
        p, _ = cell.pose(name)
        if float(p[2]) < FLOOR_Z:
            on_floor.append(name)
    rude = sorted(set(mon.get("rude", [])))
    charge = PENALTY["broken"] * len(broken) + PENALTY["on_floor"] * len(on_floor) + PENALTY["rude"] * len(rude)
    earned = float(sum(credit.values()))
    total = max(0.0, earned - charge)
    return {
        "stage_score": total / TOTAL,
        "credit": total,
        "credit_earned": earned,
        "penalty": charge,
        "damage": {"broken": broken, "on_floor": sorted(on_floor), "rude": rude},
        "credit_by_part": {k: round(v, 4) for k, v in credit.items()},
        "spill_parts": {k: round(v, 4) for k, v in spill.items()},
        "closing_parts": close,
        "tables": detail["tables"],
        "turned": turned,
        "spill": {"t": t_sp, "table": cell.layout["spill"]["table"], "removed_fraction": round(removed, 4),
                  "dry_at": sp.dry_at, "cone_placed_at": mon.get("cone_placed_at"),
                  "cone_left_at": mon.get("cone_left_at"), "cone_home_end": cell.shift.cone_home()},
        "delivered": len(delivered), "guest_items": len(all_items),
        "events": list(cell.shift.events),
        "crew_visits": list(cell.shift.crew_visits),
        "process": pr.record(),
        "grip_events": [list(e) for e in cell.grips.events][:600],
        "clip_events": [list(e) for e in cell.clips.events][:100],
        "broken": dict(cell.fragile.broken),
        "safety_ok": cell.safety.ok,
        "violations": list(cell.safety.violations),
        "safety_peaks": dict(cell.safety.peaks),
        "safety_peak_who": dict(cell.safety.peak_who),
        "sim_time_s": cell.sim_time,
        "camera_frames": dict(cell.frames),
        "camera_frames_total": cell.frames_total,
        "render_wall_s": cell.render_wall,
        "done_reason": cell.done_reason,
    }
