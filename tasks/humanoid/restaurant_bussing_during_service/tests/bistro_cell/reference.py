"""The privileged reference busser (the oracle's route; grader-side only).

It drives the robot through the cell's own controller (the same `command` / `step` a program uses) but reads the true
state: the objects' poses, which parties are seated, the spill, the soil, the sanitizer coverage, the crumbs. The
shift, in the order a busser works it:
  * a spill (as soon as it happens): the wet-floor cone beside it, the mop over it until it is up, the mop back in its
    clip; once the floor is dry, the cone back on its spot;
  * a table whose party has left (in the order they left): the bus cart pushed along the aisle to the table's side
    and left there; the guests' things carried to the tub two at a time (plates stacked in the tub's two stacks,
    glasses in its corners, cutlery and napkins on top); the cloth from the sanitizer bucket, the top wiped and
    sanitized from both ends of the table, the crumbs pulled into the crumb pan held at the edge; the cloth back in
    the bucket; a roll-up at every seat from the caddy;
  * the cart to the dock when the tub holds a table's things or the caddy runs short, the crew empties it;
  * at the shift's end (or when nothing is left to do) the cart back home.

Motion layer: A* on a 5 cm grid of the floor for the robot alone (walk), straight pushes along the aisles and turns in
the open east zone for the robot with the cart (cart_*); arm moves are grasp-site pose targets solved by the cell's IK
and checked on a scratch copy of the state for the robot's and the held object's penetrations (Checker).
"""

from __future__ import annotations

import heapq
import math

import mujoco
import numpy as np

from . import objects as O
from . import room as RM
from .natural_ik import REACH_SEEDS, NaturalIK
from .robot import ARM_HOME, PELVIS_HEIGHT, LegPosture

TICK = 0.01
GRID = 0.05
ROBOT_R = 0.32            # m: the walking robot's footprint radius (arms at home)
CARRY_R = 0.36            # ... carrying things in front of the chest
WORK_GAP = 0.19           # m: the pelvis this far from a table's edge when working at it
WORK_H, WORK_PITCH = 0.70, 0.45
TUCK = {"left": [0.20, 0.22, 0.0, 0.45, 0.0, 0.0, 0.0], "right": [0.20, -0.22, 0.0, 0.45, 0.0, 0.0, 0.0]}
# hands raised in front of the chest (natural IK of the grasp site at (0.32, +-0.20, 0.98) in the base frame)
READY = {"left": [-0.04, 0.37, 0.12, -0.66, 1.21, 0.14, -0.70], "right": [-0.04, -0.37, -0.12, -0.66, -1.21, 0.14, 0.70]}
# ... above the tub's rim, for work at the cart (the grasp site at (0.25, +-0.20, 1.15))
READY_HIGH = {"left": [-1.02, 1.63, -0.09, -0.79, 0.09, 1.03, -1.26],
              "right": [-1.02, -1.63, 0.09, -0.79, -0.09, 1.03, 1.26]}


# ================================================================================================ geometry helpers
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def frame(approach, pinch):
    """A grasp-site rotation: x = approach, y = pinch, z = x cross y (both given in the room, made orthonormal)."""
    a = np.asarray(approach, float)
    a = a / np.linalg.norm(a)
    p = np.asarray(pinch, float)
    p = p - (p @ a) * a
    p = p / np.linalg.norm(p)
    return np.stack([a, p, np.cross(a, p)], axis=1)


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


# ================================================================================================ the floor planner
def floor_boxes(include_cart=None, extra=()):
    """What stands on the floor (x0, y0, x1, y1): tables, chairs, the station, the pass, the host stand, the planters,
    the mop bucket; plus `extra` (the cart where it stands, the cone)."""
    out = []
    for name, t in RM.tables().items():
        (cx, cy), (sx, sy) = t["center"], t["size"]
        out.append((cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2))
        for st in t["seats"]:
            x, y = st["chair"]
            out.append((x - 0.25, y - 0.25, x + 0.25, y + 0.25))
    b = RM.STATION["box"]
    out.append((b[0], b[1], b[3], b[4]))
    b = RM.PASS["counter"]
    out.append((b[0], b[1], b[3], b[4]))
    b = RM.HOST_STAND["box"]
    out.append((b[0], b[1], b[3], b[4]))
    for pl in RM.PLANTERS.values():
        (x, y), r = pl["center"], pl["foliage_r"] * 0.8
        out.append((x - r, y - r, x + r, y + r))
    (mx, my), (bx, by, _bz) = RM.MOP_HOME["bucket"], RM.MOP_HOME["bucket_size"]
    out.append((mx - bx / 2, my - by / 2, RM.ROOM_X, my + by / 2 + 0.05))
    out += list(extra)
    return out


class Planner:
    """A* on a 5 cm occupancy grid of the floor, obstacles inflated by a footprint radius."""

    def __init__(self, radius=ROBOT_R):
        self.nx, self.ny = int(round(RM.ROOM_X / GRID)), int(round(RM.ROOM_Y / GRID))
        self.radius = radius
        xs = (np.arange(self.nx) + 0.5) * GRID
        ys = (np.arange(self.ny) + 0.5) * GRID
        self.X, self.Y = np.meshgrid(xs, ys, indexing="ij")
        self.static = self._inflate(floor_boxes(), radius)

    def _inflate(self, boxes, r):
        X, Y = self.X, self.Y
        occ = (X < r) | (Y < r) | (X > RM.ROOM_X - r) | (Y > RM.ROOM_Y - r)
        for x0, y0, x1, y1 in boxes:
            ddx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
            ddy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
            occ |= (ddx * ddx + ddy * ddy) < r * r
        return occ

    def cell(self, x, y):
        return (int(np.clip(x / GRID, 0, self.nx - 1)), int(np.clip(y / GRID, 0, self.ny - 1)))

    def path(self, start, goal, extra=(), free_r=0.25):
        occ = self.static.copy()
        if extra:
            occ |= self._inflate(extra, self.radius) & ~self._inflate([], self.radius)
        s, g = self.cell(*start), self.cell(*goal)
        I, J = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        k = int(round(free_r / GRID))
        near = (np.hypot(I - s[0], J - s[1]) <= k) | (np.hypot(I - g[0], J - g[1]) <= k)
        hard = self._inflate(floor_boxes(extra=extra), 0.22)
        occ = np.where(near, hard, occ)
        occ[s] = occ[g] = False
        openq = [(0.0, s)]
        came, cost = {s: None}, {s: 0.0}
        nb = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414),
              (-1, -1, 1.414)]
        while openq:
            _, c = heapq.heappop(openq)
            if c == g:
                break
            for di, dj, w in nb:
                n = (c[0] + di, c[1] + dj)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or occ[n]:
                    continue
                nc = cost[c] + w
                if nc < cost.get(n, 1e18):
                    cost[n] = nc
                    came[n] = c
                    heapq.heappush(openq, (nc + math.hypot(g[0] - n[0], g[1] - n[1]), n))
        if g not in came:
            return None
        cells = []
        c = g
        while c is not None:
            cells.append(c)
            c = came[c]
        cells.reverse()
        pts = [((i + 0.5) * GRID, (j + 0.5) * GRID) for i, j in cells]
        pts[0], pts[-1] = tuple(start), tuple(goal)
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._clear(pts[i], pts[j], occ):
                j -= 1
            out.append(pts[j])
            i = j
        return out[1:]

    def _clear(self, a, b, occ):
        n = int(max(2, math.hypot(b[0] - a[0], b[1] - a[1]) / (GRID / 2)))
        for t in np.linspace(0, 1, n):
            i, j = self.cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if occ[i, j]:
                return False
        return True


# ================================================================================================ collision checking
class Checker:
    """A scratch copy of the state for checking robot configurations: the robot's links (not the hands) and what the
    hands hold must not penetrate the room, the furniture, the guests or the loose objects (except the ones allowed)."""

    def __init__(self, cell):
        from .robot import ROBOT
        self.c = cell
        self.m = cell.m
        self.d = mujoco.MjData(self.m)
        self.r = cell.robot
        self.legs = LegPosture(cell.m, cell.robot)
        m = self.m
        self.robot_geoms = set(int(g) for g in self.r.geom_ids if m.geom_contype[g] & ROBOT)
        hb = set()
        for side in ("left", "right"):
            hb |= set(int(b) for b in self.r.hand_bodies[side])
        self.hand_geoms = {g for g in self.robot_geoms if int(m.geom_bodyid[g]) in hb}
        self.robot_bodies = set(int(b) for b in self.r.body_ids)

    def set(self, base, h, pitch, yaw_w=0.0, arm_q=None):
        """The robot at stance (base, stand height h, waist pitch / yaw): the robot's measured state where that is
        the stance it stands in now (a plan for here is exact), else the modelled posture."""
        d, r = self.d, self.r
        cd = self.c.d
        d.qpos[:] = cd.qpos
        b_now = cd.qpos[r.base_q[:3]]
        h_now = PELVIS_HEIGHT + float(cd.qpos[r.base_q[3]])
        w_now = cd.qpos[r.waist_q]
        here = (abs(float(base[0]) - b_now[0]) < 0.01 and abs(float(base[1]) - b_now[1]) < 0.01
                and abs(math.remainder(float(base[2]) - b_now[2], 2 * math.pi)) < 0.01 and abs(h - h_now) < 0.005
                and abs(pitch - w_now[2]) < 0.01 and abs(yaw_w - w_now[0]) < 0.01)
        if not here:
            d.qpos[r.base_q[:3]] = base
            d.qpos[r.base_q[3]] = h - PELVIS_HEIGHT
            legs = self.legs.targets(h)
            for s in ("left", "right"):
                d.qpos[r.leg_q[s]] = legs
            d.qpos[r.waist_q] = [yaw_w, 0.0, pitch]
        if arm_q:
            for s, q in arm_q.items():
                d.qpos[r.arm_q[s]] = q

    def place_held(self, side, name, rel):
        """Put a held object where the hand holds it (rel = its pose in the grasp-site frame)."""
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        sid = self.r.grasp_site[side]
        ps, Rs = d.site_xpos[sid], d.site_xmat[sid].reshape(3, 3)
        pl, Rl = rel
        Ro = Rs @ Rl
        po = ps + Rs @ pl
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
        a = m.jnt_qposadr[j]
        d.qpos[a:a + 3] = po
        d.qpos[a + 3:a + 7] = quat_of(Ro)

    def bad(self, held=(), allow=(), tol=0.004):
        """Penetrations (body names) of the robot's links (not the hands) and of the held objects."""
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        sub = self.c.grips.subtree
        held_b, allow_b = set(), set()
        for n in held:
            held_b |= sub.get(n) or {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)}
        for n in allow:
            allow_b |= sub.get(n) or {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)}
        out = []
        for i in range(d.ncon):
            ct = d.contact[i]
            if ct.dist > -tol:
                continue
            g1, g2 = int(ct.geom1), int(ct.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            for ga, gb, ba, bb in ((g1, g2, b1, b2), (g2, g1, b2, b1)):
                if bb in self.robot_bodies or bb in held_b or bb in allow_b:
                    continue
                if (ga in self.robot_geoms and ga not in self.hand_geoms) or ba in held_b:
                    out.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, ba),
                                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bb) or "world", round(float(ct.dist), 4)))
                    break
        return out


# ================================================================================================ the reference
class Ref:
    def __init__(self, cell, log=None):
        self.c = cell
        self.m, self.d, self.r = cell.m, cell.d, cell.robot
        self.log = log or (lambda s: None)
        self.walk_planner = Planner(ROBOT_R)
        self.carry_planner = Planner(CARRY_R)
        self.chk = Checker(cell)
        self.nik = NaturalIK(cell.m, cell.robot)
        self.held = {"left": None, "right": None}
        self.rel = {}                     # held name -> (pos, R) of the object in the grasp-site frame
        self.notes = []
        self.trace = False
        self.trace_ticks = 0
        self._consolidated = set()
        self.stats = {"picks": 0, "pick_fail": 0, "places": 0, "place_fail": 0, "walks": 0, "walk_fail": 0}

    # ------------------------------------------------------------------------------------------ state
    def step(self, n=1):
        if not self.c.done:
            self.c.step(int(max(1, n)))

    @property
    def t(self):
        return self.c.sim_time

    def out_of_time(self, margin=5.0):
        return self.c.done or self.t > self.c.sim_cap - margin

    def base(self):
        return np.array(self.d.qpos[self.r.base_q[:3]], float)

    def site(self, side):
        sid = self.r.grasp_site[side]
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def obj(self, name):
        b = self.c.body[name]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def body_push(self):
        """The largest contact force between a robot link (not a hand) and anything else now (N)."""
        m, d = self.m, self.d
        rb = self.chk.robot_bodies
        hand = self.chk.hand_geoms
        worst = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            ct = d.contact[i]
            g1, g2 = int(ct.geom1), int(ct.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if (b1 in rb) == (b2 in rb):
                continue
            if g1 in hand or g2 in hand:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            worst = max(worst, abs(float(f6[0])))
        return worst

    # ------------------------------------------------------------------------------------------ body
    def posture(self, h, pitch, yaw_w=0.0, timeout=4.0):
        """Stand height and waist (yaw, pitch) on a smooth ramp, waiting until the body is there."""
        ctrl = self.c.ctrl
        h0, p0, y0 = float(ctrl.stand_cmd), float(ctrl.waist_cmd[2]), float(ctrl.waist_cmd[0])
        dh, dp, dy = float(h) - h0, float(pitch) - p0, float(yaw_w) - y0
        T = max(math.pi / 2 * abs(dh) / (0.9 * 0.15), math.pi / 2 * max(abs(dp), abs(dy)) / (0.9 * 0.8), 0.02)
        hold = {s: self.site(s) for s in ("left", "right") if self.held[s]}
        if hold:
            T = max(2.0 * T, 0.5)
        n = int(math.ceil(T / (2 * TICK)))
        for k in range(1, n + 1):
            u = 0.5 - 0.5 * math.cos(math.pi * k / n)
            cmd = {"stand_height": h0 + dh * u, "waist": [y0 + dy * u, 0.0, p0 + dp * u]}
            self.c.command(cmd)
            if hold:
                self._hold(hold, h0 + dh * u, p0 + dp * u, y0 + dy * u)
            self.step(2)
        self.c.command({"stand_height": float(h), "waist": [float(yaw_w), 0.0, float(pitch)]})
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            if abs(ctrl.stand - ctrl.stand_cmd) < 1e-4 and float(np.max(np.abs(ctrl.waist - ctrl.waist_cmd))) < 1e-4:
                wq = self.d.qpos[self.r.waist_q]
                if float(np.max(np.abs(wq - ctrl.waist_cmd))) < 0.01:
                    return True
        return False

    def _hold(self, hold, h, pitch, yaw_w):
        """Arm joints that keep each holding hand at its room pose while the body moves (joint targets)."""
        b = self.base()
        self.chk.set(b, h, pitch, yaw_w)
        for s, (p, R) in hold.items():
            q, ep, er = self.c.ctrl.ik.solve(self.chk.d.qpos, s, p, quat_of(R), self.d.qpos[self.r.arm_q[s]].copy(),
                                             iters=20)
            if ep < 0.01:
                self.c.command({f"{s}_arm": {"q": [float(v) for v in q]}})

    def goto(self, x, y, yaw, tol=0.02, yaw_tol=0.03, timeout=40.0):
        self.c.command({"base_goal": [float(x), float(y), float(yaw)]})
        prev = self.base()
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            b = self.base()
            if math.hypot(b[0] - x, b[1] - y) < tol and abs(wrap(b[2] - yaw)) < yaw_tol:
                break
            if self.c.done:
                break
            if self.body_push() > 60.0:
                self.log("[ref] goto: pressing %.0f N at (%.2f, %.2f): backing off" % (self.body_push(), b[0], b[1]))
                dv = prev[:2] - b[:2]
                n = float(np.linalg.norm(dv))
                dv = dv / n * 0.06 if n > 1e-4 else -0.06 * np.array([math.cos(b[2]), math.sin(b[2])])
                self.c.command({"base_goal": [float(b[0] + dv[0]), float(b[1] + dv[1]), float(b[2])]})
                self.step(50)
                return False
            prev = b
        self.step(3)
        b = self.base()
        return math.hypot(b[0] - x, b[1] - y) < max(tol * 2, 0.04)

    def cart_box(self, margin=0.03):
        """The parked cart's footprint (its corners' bounding box) as a floor obstacle."""
        pts = np.array(self.c.cart.corners())
        return (float(pts[:, 0].min()) - margin, float(pts[:, 1].min()) - margin, float(pts[:, 0].max()) + margin,
                float(pts[:, 1].max()) + margin)

    def obstacles(self, cart=True):
        out = []
        if cart and not self.c.cart.grasp:
            out.append(self.cart_box())
        po, _ = self.obj("cone")
        if po[2] < 0.1 and self.held["left"] != "cone" and self.held["right"] != "cone":
            out.append((po[0] - 0.17, po[1] - 0.17, po[0] + 0.17, po[1] + 0.17))
        return out

    def back_off(self, dist=0.28):
        """Step straight back from where the robot works (before it turns: its arms would sweep the chairs)."""
        b = self.base()
        return self.goto(b[0] - dist * math.cos(b[2]), b[1] - dist * math.sin(b[2]), b[2], tol=0.03, yaw_tol=0.05)

    def approach(self, x, y, yaw, back=0.35, extra=(), h=None, pitch=None, high=False):
        """Walk to a stance by a fixture: to a point `back` metres behind it (planned), take the working posture
        there (h, pitch: the free hands raised first), then straight in."""
        bx, by = x - back * math.cos(yaw), y - back * math.sin(yaw)
        b = self.base()
        if math.hypot(b[0] - x, b[1] - y) > 0.05:
            ok = self.walk(bx, by, yaw, extra=extra)
            if not ok:
                return False
        if h is not None:
            self.ready([s for s in ("left", "right") if not self.held[s]], high=high)
            self.posture(h, pitch if pitch is not None else 0.0)
        return self.goto(x, y, yaw, tol=0.012, yaw_tol=0.015)

    def sync_held(self):
        """Bring the hands' record in line with the grips (a slip is logged); True if nothing was lost."""
        ok = True
        for s in ("left", "right"):
            h = self.c.grips.holding(s)
            if self.held[s] and h != self.held[s]:
                self.log("[ref] the %s hand lost %s at t %.1f" % (s, self.held[s], self.t))
                self.rel.pop(self.held[s], None)
                self.held[s] = h
                ok = False
        return ok

    def walk(self, x, y, yaw, extra=(), carrying=None):
        """Walk to (x, y, yaw): the planned path's waypoints, facing the way it goes, the last turn in place."""
        self.stats["walks"] += 1
        carrying = any(self.held.values()) if carrying is None else carrying
        pl = self.carry_planner if carrying else self.walk_planner
        extra = list(extra) + self.obstacles()
        b = self.base()
        if math.hypot(b[0] - x, b[1] - y) < 0.03:
            return self.goto(x, y, yaw, tol=0.015)
        path = pl.path((b[0], b[1]), (x, y), extra=extra)
        if path is None:
            path = self.walk_planner.path((b[0], b[1]), (x, y), extra=extra)
        if path is None:
            # (the start wedged by something it stands at: a step back first, else a step the way that is clear)
            self.back_off(0.25)
            b = self.base()
            path = self.walk_planner.path((b[0], b[1]), (x, y), extra=extra)
            if path is None:
                for a in np.linspace(0.0, 2 * math.pi, 12, endpoint=False):
                    for dist in (0.25, 0.40):
                        p_ = b[:2] + dist * np.array([math.cos(b[2] + a), math.sin(b[2] + a)])
                        if self._clear(p_, ROBOT_R - 0.04):
                            self.goto(p_[0], p_[1], b[2], tol=0.04, yaw_tol=0.1, timeout=8.0)
                            b = self.base()
                            path = self.walk_planner.path((b[0], b[1]), (x, y), extra=extra)
                            break
                    if path is not None:
                        break
        if path is None:
            self.stats["walk_fail"] += 1
            self.log("[ref] walk: no path to (%.2f, %.2f)" % (x, y))
            return False
        self.pursue(path)
        # the rest (the last bit, or all of it after a stall) waypoint by waypoint on a path planned from here: never
        # a straight line to the goal (it would cut through a chair)
        b = self.base()
        if math.hypot(b[0] - x, b[1] - y) > 0.35:
            rest = pl.path((b[0], b[1]), (x, y), extra=extra) or self.walk_planner.path((b[0], b[1]), (x, y),
                                                                                        extra=extra)
            if rest is None:
                self.stats["walk_fail"] += 1
                self.log("[ref] walk: no path on from (%.2f, %.2f) to (%.2f, %.2f)" % (b[0], b[1], x, y))
                return False
            if not self._waypoints(rest, x, y, yaw) and self.unstick():
                b = self.base()
                rest = self.walk_planner.path((b[0], b[1]), (x, y), extra=extra)
                if rest is None or not self._waypoints(rest, x, y, yaw):
                    self.stats["walk_fail"] += 1
                    return False
        ok = self.goto(x, y, yaw, tol=0.012, yaw_tol=0.012)
        self.sync_held()
        return ok

    def unstick(self, dist=0.22):
        """Step straight away from the nearest footprint on the floor (a robot pressed against a chair, a table or the
        cart: every planned walk from there would push on into it)."""
        b = self.base()
        best = None
        for (x0, y0, x1, y1) in floor_boxes(extra=self.obstacles()):
            qx, qy = min(max(b[0], x0), x1), min(max(b[1], y0), y1)
            dd = math.hypot(b[0] - qx, b[1] - qy)
            if best is None or dd < best[0]:
                best = (dd, qx, qy)
        if best is None or best[0] > ROBOT_R + 0.05:
            return False
        _, qx, qy = best
        v = np.array([b[0] - qx, b[1] - qy])
        n = float(np.linalg.norm(v))
        v = v / n if n > 1e-6 else -np.array([math.cos(b[2]), math.sin(b[2])])
        self.log("[ref] unstick: %.2f m from a footprint at (%.2f, %.2f): stepping away" % (best[0], qx, qy))
        self.goto(b[0] + v[0] * dist, b[1] + v[1] * dist, b[2], tol=0.04, yaw_tol=0.2, timeout=6.0)
        return True

    def _waypoints(self, path, x, y, yaw):
        """Follow the path's waypoints one goal at a time (turning in place where it bends), up to its last."""
        prev = tuple(self.base()[:2])
        for k, (px, py) in enumerate(path[:-1]):
            hd = math.atan2(py - prev[1], px - prev[0]) if math.hypot(px - prev[0], py - prev[1]) > 0.05 else \
                self.base()[2]
            if abs(wrap(hd - self.base()[2])) > 0.35:
                self.goto(prev[0], prev[1], hd, tol=0.05, yaw_tol=0.08)
            if not self.goto(px, py, hd, tol=0.06, yaw_tol=0.15):
                return False
            prev = (px, py)
        px, py = path[-1]
        d_ = math.hypot(px - prev[0], py - prev[1])
        if d_ > 0.4:
            hd = math.atan2(py - prev[1], px - prev[0])
            if abs(wrap(hd - self.base()[2])) > 0.35:
                self.goto(prev[0], prev[1], hd, tol=0.05, yaw_tol=0.08)
            self.goto(px, py, hd, tol=0.03, yaw_tol=0.1)
        return True

    def pursue(self, path, look=0.15, v=0.5, stop_at=0.30):
        """Drive along the planned path with velocity commands (pure pursuit: the heading turned towards a point
        `look` metres ahead on the path, slower in turns, turning in place when it points far off), until `stop_at`
        metres before its end (the last bit is the goal approach's). Stops early if the base stalls."""
        b = self.base()
        P = np.array([b[:2]] + [tuple(p) for p in path], float)
        if len(P) < 2:
            return
        d = np.diff(P, axis=0)
        segl = np.hypot(d[:, 0], d[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(segl)])
        total = float(cum[-1])
        if total < stop_at + 0.05:
            return

        def point_at(sv):
            i = int(np.clip(np.searchsorted(cum, sv, side="right") - 1, 0, len(segl) - 1))
            u = (sv - cum[i]) / max(1e-9, segl[i])
            return P[i] + d[i] * float(np.clip(u, 0.0, 1.0))
        s_prev = 0.0
        stall_t, stall_p = self.t, self.base()[:2].copy()
        while not self.c.done:
            b = self.base()
            # the progress along the path (searched near the last one: the path may pass close to itself)
            best = (1e9, s_prev)
            for i in range(len(segl)):
                if cum[i + 1] < s_prev - 0.3 or cum[i] > s_prev + 1.0:
                    continue
                a_, dv = P[i], d[i]
                u = float(np.clip(((b[:2] - a_) @ dv) / max(1e-9, segl[i] ** 2), 0.0, 1.0))
                q = a_ + dv * u
                dist = float(np.hypot(*(b[:2] - q)))
                if dist < best[0]:
                    best = (dist, float(cum[i] + u * segl[i]))
            s_prev = max(s_prev, best[1])
            rem = total - s_prev
            if rem <= stop_at:
                break
            tgt = point_at(min(total, s_prev + look))
            hd = math.atan2(tgt[1] - b[1], tgt[0] - b[0])
            err = wrap(hd - b[2])
            if abs(err) > 0.4:
                vx = 0.0                     # (turn first: driving on, it would cut the corner into a chair)
            else:
                vx = min(v, 0.10 + 0.40 * math.cos(err) ** 8, 0.9 * rem + 0.1)
            wz = float(np.clip(2.2 * err, -0.8, 0.8))
            self.c.command({"base_velocity": [float(vx), 0.0, wz]})
            self.step(5)
            if self.body_push() > 60.0:
                self.log("[ref] pursue: pressing %.0f N at %s: stop" % (self.body_push(), np.round(self.base(), 2)))
                break
            if vx < 0.05:
                # (turning in place: not a stall)
                stall_t, stall_p = self.t, self.base()[:2].copy()
            elif self.t - stall_t > 2.5:
                if float(np.hypot(*(self.base()[:2] - stall_p))) < 0.03:
                    self.log("[ref] pursue: the base stalls at %s" % np.round(self.base(), 2))
                    break
                stall_t, stall_p = self.t, self.base()[:2].copy()
        self.c.command({"base_velocity": [0.0, 0.0, 0.0]})
        self.step(2)

    # ------------------------------------------------------------------------------------------ arms and hands
    def to_base(self, p, R):
        b = self.base()
        Rb = Rz(b[2])
        pb = Rb.T @ (np.asarray(p, float) - np.array([b[0], b[1], 0.0]))
        return pb, (None if R is None else Rb.T @ np.asarray(R, float))

    def seg_done(self, side):
        """The arm controller's present pose segment has run its course (its interpolation is at the target)."""
        sg = self.c.ctrl.seg.get(side)
        return sg is None or sg[3] >= sg[2] - 1e-9

    def arm_to(self, targets, path="line", timeout=8.0, tol=0.004, rtol=0.03, frame="room"):
        """Grasp-site pose targets given in the room {side: (p, R)} (R None: position only), waiting until each hand
        is within tol of its target and its controller segment is done. frame="base": the targets are sent in the
        robot's base frame (they travel with the base: a carry), converted at the present base pose."""
        cmd = {}
        for s, (p, R) in targets.items():
            pp, RR = (p, R) if frame == "room" else self.to_base(p, R)
            if RR is None:
                cmd[f"{s}_arm"] = {"pos": [float(v) for v in pp], "frame": frame, "path": path}
            else:
                cmd[f"{s}_arm"] = {"pose": [float(v) for v in pp] + [float(v) for v in quat_of(RR)], "frame": frame,
                                   "path": path}
        self.c.command(cmd)
        ok = False
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            ok = True
            for s, (p, R) in targets.items():
                ps, Rs = self.site(s)
                if np.linalg.norm(ps - np.asarray(p)) > tol or not self.seg_done(s):
                    ok = False
                if R is not None:
                    e = Rs.T @ np.asarray(R)
                    if math.acos(float(np.clip((np.trace(e) - 1) / 2, -1, 1))) > rtol:
                        ok = False
            if ok or self.c.done:
                break
        return ok

    def arm_q(self, side, q, timeout=6.0, tol=0.01):
        self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            if float(np.max(np.abs(self.d.qpos[self.r.arm_q[side]] - np.asarray(q)))) < tol:
                return True
        return False

    def ready(self, sides=("left", "right"), high=False):
        """Free hands raised in front of the chest (before the body bends or squats near a table or the cart: hands
        hanging by the hips would swing forward into its edge); high: above the tub's rim (at the cart)."""
        tbl = READY_HIGH if high else READY
        for s in sides:
            if self.held[s]:
                continue
            q0 = self.d.qpos[self.r.arm_q[s]].copy()
            tgt = np.asarray(tbl[s], float)
            if float(np.max(np.abs(q0 - tgt))) <= 0.05:
                continue
            # (a checked joint path, through the raised posture if the straight one is blocked; else the arm stays)
            path = self.joint_path(s, q0, tgt)
            if path is None and not high:
                a = self.joint_path(s, q0, READY_HIGH[s])
                b = self.joint_path(s, READY_HIGH[s], tgt) if a is not None else None
                path = a + b if a is not None and b is not None else None
            if path is None:
                self.log("[ref] ready %s: no free path: the arm stays" % s)
                continue
            self.run_joints(s, path, tol=0.05)

    def home(self, sides=("left", "right"), tuck=True):
        """Free arms to the tucked pose (hanging by the body: nothing in front of it) or the controller's home, along
        checked joint paths (through the raised ready posture if the straight one is blocked)."""
        for s in sides:
            if self.held[s]:
                continue
            tgt = np.array(TUCK[s] if tuck else ARM_HOME[s], float)
            q0 = self.d.qpos[self.r.arm_q[s]].copy()
            if float(np.max(np.abs(q0 - tgt))) < 0.02:
                continue
            # (an arm link already grazing a movable thing, the tub's rim say, may leave it: the first leg of the
            # path allows what it touches now, never the room's fixed things)
            now = self.check_pose(s, q0, tol=0.003)
            esc = tuple(sorted({o for (_l, o, _d) in now if o != "world"})) if now else ()
            if now and len(esc) < len({o for (_l, o, _d) in now}):
                esc = ()
            path = None
            for vias in ((), (READY_HIGH[s],), (READY[s],), (READY_HIGH[s], READY[s])):
                pts = [q0] + [np.array(v, float) for v in vias] + [tgt]
                segs = []
                for k, (qa, qb) in enumerate(zip(pts[:-1], pts[1:])):
                    seg = self.joint_path(s, qa, qb, allow=esc if k == 0 else ())
                    if seg is None:
                        segs = None
                        break
                    segs += seg
                if segs is not None:
                    path = segs
                    break
            if path is None:
                self.log("[ref] home %s: no free path: the arm stays (touching now: %s)" % (s, now))
                continue
            self.run_joints(s, path, tol=0.05)

    def hand(self, side, close, settle=25):
        self.c.command({f"{side}_hand": "close" if close else "open"})
        self.step(settle)
        name = self.c.grips.holding(side)
        if close:
            self.held[side] = name
            if name and name != "cart":
                ps, Rs = self.site(side)
                po, Ro = self.obj(name)
                self.rel[name] = (Rs.T @ (po - ps), Rs.T @ Ro)
            return name
        old = self.held[side]
        self.held[side] = None
        if old and old not in self.held.values():
            self.rel.pop(old, None)
        return old

    def ik(self, side, p, R, base=None, h=None, pitch=None, yaw_w=0.0, seed=None, iters=60):
        """(q, position error, rotation error) of the grasp-site pose with the body at (base, h, pitch, yaw_w): the
        natural-posture IK from the seed (the arm's present joints), then from a few natural reaching postures."""
        b = self.base() if base is None else base
        h = self.c.ctrl.stand_cmd if h is None else h
        pitch = self.c.ctrl.waist_cmd[2] if pitch is None else pitch
        self.chk.set(b, h, pitch, yaw_w)
        q0 = self.d.qpos[self.r.arm_q[side]].copy() if seed is None else np.asarray(seed, float)
        quat = None if R is None else quat_of(R)
        pp = np.asarray(p, float)
        best = None
        for k, sd in enumerate([q0, np.array(ARM_HOME[side], float)] + [np.array(v, float) for v in REACH_SEEDS[side]]):
            q, ep, er = self.nik.solve(self.chk.d.qpos, side, pp, quat, sd, iters=iters, natural=True)
            sc = ep + 0.3 * er
            if best is None or sc < best[0] - 1e-6:
                best = (sc, q, ep, er)
            if ep < 0.003 and (R is None or er < 0.03):
                break
        return best[1], best[2], best[3]

    def ik_soft(self, side, p, R, seed=None, w_rot=0.12):
        """IK with the position first and the rotation weighted lightly (pressing a tool flat: a few degrees of tilt
        do no harm, a centimetre of position does)."""
        q0 = self.d.qpos[self.r.arm_q[side]].copy() if seed is None else np.asarray(seed, float)
        self.chk.set(self.base(), self.c.ctrl.stand_cmd, self.c.ctrl.waist_cmd[2], self.c.ctrl.waist_cmd[0])
        q, ep, er = self.nik.solve(self.chk.d.qpos, side, np.asarray(p, float), quat_of(R), q0, iters=40,
                                   w_rot=w_rot, natural=True)
        if ep > 0.004:
            q2, ep2, er2 = self.ik(side, p, R, seed=q0)
            if ep2 < ep:
                return q2, ep2, er2
        return q, ep, er

    def check_pose(self, side, q, held=(), allow=(), base=None, h=None, pitch=None, yaw_w=0.0, other_q=None,
                   tol=0.004):
        b = self.base() if base is None else base
        h = self.c.ctrl.stand_cmd if h is None else h
        pitch = self.c.ctrl.waist_cmd[2] if pitch is None else pitch
        arm_q = {side: q}
        if other_q is not None:
            arm_q.update(other_q)
        self.chk.set(b, h, pitch, yaw_w, arm_q=arm_q)
        riders = []
        for s in ("left", "right"):
            n = self.held[s]
            if n and n != "cart" and n in self.rel:
                self.chk.place_held(s, n, self.rel[n])
                rs = self.riders(n)
                if rs:
                    # (what rides on a held plate, the cutlery or a napkin on it, goes where the plate goes: carried
                    # with it in the check, so it clears the tub's rim too; a light napkin squeezed against a rim
                    # is flung off it)
                    ps_, Rs_ = self.site(s)
                    for r_ in rs:
                        pr_, Rr_ = self.obj(r_)
                        self.chk.place_held(s, r_, (Rs_.T @ (pr_ - ps_), Rs_.T @ Rr_))
                riders += rs
        return self.chk.bad(held=[n for n in self.held.values() if n and n != "cart"] + list(held) + riders,
                            allow=tuple(allow), tol=tol)

    # ------------------------------------------------------------------------------------------ grasps
    def kind(self, name):
        return self.c.meta["kinds"][name]

    def grasps(self, name, side, base=None):
        """Candidate grasps of `name` as it lies now for the `side` hand: [(site p, R, pre-grasp offset (room),
        lift offset)], best first (nearest the robot, the hand's natural side)."""
        kind = self.kind(name)
        po, Ro = self.obj(name)
        b = self.base() if base is None else np.asarray(base, float)
        to_robot = np.array([b[0] - po[0], b[1] - po[1], 0.0])
        nr = float(np.linalg.norm(to_robot))
        to_robot = to_robot / nr if nr > 1e-6 else np.array([-math.cos(b[2]), -math.sin(b[2]), 0.0])
        lat = np.array([-math.sin(b[2]), math.cos(b[2]), 0.0]) * (1.0 if side == "left" else -1.0)
        z = np.array([0.0, 0.0, 1.0])
        out = []
        up = np.array([0.0, 0.0, 0.06])
        if kind == "plate":
            R_, hh = O.PLATE["r"], O.PLATE["h"]
            c = po + Ro @ np.array([0.0, 0.0, hh / 2])
            base_ang = math.atan2(to_robot[1], to_robot[0])
            for da in (0.35, 0.0, 0.7, -0.35, 1.0, -0.7):
                a = base_ang + da * (1.0 if side == "left" else -1.0)
                u = np.array([math.cos(a), math.sin(a), 0.0])
                p = c + u * (R_ - 0.022)
                Rg = frame(-u, z if side == "left" else -z)
                out.append((p, Rg, u * 0.07 + np.array([0, 0, 0.01]), up))
        elif kind == "glass":
            c = po + Ro @ np.array([0.0, 0.0, 0.06])
            base_ang = math.atan2(to_robot[1], to_robot[0])
            for da in (0.0, 0.4, -0.4, 0.8):
                a = base_ang + da
                u = np.array([math.cos(a), math.sin(a), 0.0])
                Rg = frame(-u, np.cross(z, u) * (1.0 if side == "left" else -1.0))
                out.append((c + u * 0.005, Rg, u * 0.08, up))
        elif kind in ("fork", "knife", "rollup"):
            ax = Ro[:, 0].copy()
            ax[2] = 0.0
            ax /= max(1e-9, np.linalg.norm(ax))
            if kind == "rollup":
                offs = (0.0, -0.06, 0.06, -0.09, 0.09)
                zc = O.ROLLUP["r"] * 1.0
            else:
                offs = (-0.055, -0.035, -0.075)
                zc = 0.004
            for off in offs:
                p = po + ax * off + np.array([0.0, 0.0, zc])
                pin = np.cross(z, ax)
                for sg in (1.0, -1.0):
                    Rg = frame(-z, pin * sg)
                    out.append((p, Rg, np.array([0.0, 0.0, 0.08]), up))
        elif kind == "napkin":
            p = po + np.array([0.0, 0.0, O.NAPKIN["half"][2] + 0.004])
            for a in np.linspace(0, math.pi, 6, endpoint=False):
                pin = np.array([math.cos(a + b[2]), math.sin(a + b[2]), 0.0])
                out.append((p, frame(-z, pin), np.array([0.0, 0.0, 0.08]), up))
        elif kind == "cloth":
            # from above, pinched across its width near one of its ends (it hangs flat under the hand and reaches
            # 14 cm beyond it)
            wy = Ro[:, 1].copy()
            wy[2] = 0.0
            wy /= max(1e-9, np.linalg.norm(wy))
            for ex in (0.03, -0.03):
                c = po + Ro @ np.array([ex, 0.0, 0.0])
                for sg in (1.0, -1.0):
                    out.append((c, frame(-z, wy * sg), np.array([0.0, 0.0, 0.05]), np.array([0.0, 0.0, 0.06])))
        elif kind in ("crumb_pan", "cone", "mop"):
            feat = next(f for f in self.c.meta["features"][name] if f[0] == "cyl")
            a = po + Ro @ np.asarray(feat[1], float)
            e = po + Ro @ np.asarray(feat[2], float)
            axv = (e - a) / np.linalg.norm(e - a)
            # (a power grasp round the bar: the pinch closes across it, the hand coming at it from the robot's side,
            # or from above where the bar lies level)
            levels = (0.5, 0.3, 0.7)
            if kind == "mop":
                # the handle upright: where it passes the hand's working height
                sz = (self.MOP_GRIP_H - a[2]) / max(1e-6, e[2] - a[2])
                levels = tuple(float(np.clip(v, 0.05, 0.95)) for v in (sz, sz - 0.05, sz + 0.05))
            for s_ in levels:
                p = a + (e - a) * s_
                aps = []
                ap = np.cross(axv, z)
                if np.linalg.norm(ap) < 0.3:
                    ap = to_robot.copy() * -1.0
                ap = ap / np.linalg.norm(ap)
                if ap @ to_robot > 0:
                    ap = -ap
                aps.append(ap)
                if abs(axv[2]) < 0.3:
                    aps.append(-z)
                for ap_ in aps:
                    pin = np.cross(axv, ap_)
                    pin = pin / max(1e-9, np.linalg.norm(pin))
                    for sg in (1.0, -1.0):
                        out.append((p, frame(ap_, pin * sg), -ap_ * 0.07, up))
        return out

    def plan_pick(self, name, side, h=None, pitch=None, base=None):
        """The first candidate grasp whose pre-grasp, grasp and lift poses the arm reaches without a collision:
        (pre, grasp, lift) as (p, R), or None."""
        riders = self.riders(name)
        for (p, R, pre, lift) in self.grasps(name, side, base=base):
            poses = [(p + pre, R), (p, R), (p + lift, R)]
            qs = []
            ok = True
            seed = None
            for k, (pp, RR) in enumerate(poses):
                q, ep, er = self.ik(side, pp, RR, base=base, h=h, pitch=pitch, seed=seed)
                if ep > 0.006 or er > 0.05:
                    if self.trace:
                        self.log("      plan_pick %s %s pose %d: ik ep %.4f er %.3f" % (name, side, k, ep, er))
                    ok = False
                    break
                oq = None
                if base is not None:
                    o = "right" if side == "left" else "left"
                    if not self.held[o]:
                        oq = {o: np.array(READY[o], float)}
                bad = self.check_pose(side, q, allow=(name,) + tuple(riders), base=base, h=h, pitch=pitch, other_q=oq)
                if bad:
                    if self.trace:
                        self.log("      plan_pick %s %s pose %d: hits %s" % (name, side, k, bad))
                    ok = False
                    break
                qs.append(q)
                seed = q
            if ok and base is not None:
                # (at a stance the robot is not at yet: the arm will start from the ready posture approach() takes)
                o = "right" if side == "left" else "left"
                oq = None if self.held[o] else {o: np.array(READY[o], float)}
                allow_ = (name,) + tuple(riders)
                # (from either ready posture: approach() takes the raised one at the cart, the other at a table)
                ok = any(self.joint_path(side, np.array(q_r[side], float), qs[0], allow=allow_, other_q=oq,
                                         base=base, h=h, pitch=pitch) is not None for q_r in (READY, READY_HIGH))
            if ok and base is None and h is None:
                q_now = self.d.qpos[self.r.arm_q[side]].copy()
                allow = (name,) + tuple(riders)
                if self.joint_path(side, q_now, qs[0], allow=allow) is None:
                    ok = any(self.joint_path(side, q_now, via, allow=allow) is not None and
                             self.joint_path(side, via, qs[0], allow=allow) is not None
                             for via in (READY_HIGH[side], READY[side], TUCK[side]))
                    if not ok and self.trace:
                        self.log("      plan_pick %s %s: no free transit" % (name, side))
            if ok:
                return poses, qs
        return None

    def riders(self, name):
        """The loose things resting on `name` (cutlery or a napkin on a plate): they go where it goes."""
        if self.kind(name) != "plate":
            return []
        po, _ = self.obj(name)
        out = []
        for n in self.c.layout["loose"]:
            if n == name or n in self.c.shift.parked or self.kind(n) not in ("fork", "knife", "napkin", "glass"):
                continue
            pn, _ = self.obj(n)
            if math.hypot(pn[0] - po[0], pn[1] - po[1]) < O.PLATE["r"] - 0.01 and 0.0 < pn[2] - po[2] < 0.05:
                out.append(n)
        return out

    def on_a_plate(self, name):
        """The plate `name` rests on, or None."""
        pn, _ = self.obj(name)
        for p_ in self.c.layout["loose"]:
            if self.kind(p_) != "plate" or p_ in self.c.shift.parked:
                continue
            pp, _ = self.obj(p_)
            if math.hypot(pn[0] - pp[0], pn[1] - pp[1]) < O.PLATE["r"] - 0.01 and 0.0 < pn[2] - pp[2] < 0.05:
                return p_
        return None

    def plate_spot(self, plate, kind):
        """Where on `plate` (its top) to set a `kind` down: the free point farthest from what already lies on it
        (a glass stands best away from the cutlery), within 5 cm of the middle for a glass, 6 cm for the rest."""
        pp, _ = self.obj(plate)
        top = pp[2] + O.PLATE["h"]
        others = [self.obj(n)[0][:2] for n in self.riders(plate)]
        best = None
        rr = 0.05 if kind == "glass" else 0.06
        for r_ in (0.0, rr * 0.5, rr):
            for a in np.linspace(0.0, 2 * math.pi, 8, endpoint=False):
                q = pp[:2] + r_ * np.array([math.cos(a), math.sin(a)])
                clear = min([float(np.linalg.norm(q - o)) for o in others] or [1.0])
                if best is None or clear > best[0] + 1e-6:
                    best = (clear, q)
        q = best[1]
        return np.array([q[0], q[1], top + 0.002]), best[0]

    def consolidate(self, T, deadline=None):
        """Before the trips to the tub: the stray cutlery on the top goes onto the plates (a plate then takes a seat's
        cutlery in one trip, as a busser stacks before carrying)."""
        plates = [n for n in self.table_items(T) if self.kind(n) == "plate"]
        if not plates:
            return
        moved = 0
        for spot in self.end_spots(T):
            base = np.array(spot)
            # (not the glasses: a glass riding on a plate topples when the plate is let go over the tub, and breaks;
            # not the napkins: a light napkin on a plate let go over the tub can wedge under it or be flung off it)
            smalls = [n for n in self.table_items(T) if self.kind(n) in ("fork", "knife")
                      and self.on_a_plate(n) is None]
            reach = [n for n in smalls if any(self.plan_pick(n, sd, h=WORK_H, pitch=WORK_PITCH, base=base)
                                              is not None for sd in ("left", "right"))]
            if not reach:
                continue
            if not self.approach(*spot, h=WORK_H, pitch=WORK_PITCH):
                continue
            reach.sort(key=lambda n: (self.kind(n) != "glass", float(np.hypot(*(self.obj(n)[0][:2] - base[:2])))))
            for n in reach:
                if self.out_of_time(60.0) or (deadline is not None and self.t > deadline):
                    break
                if self.on_a_plate(n) is not None or n not in self.table_items(T):
                    continue
                p0, R0 = self.obj(n)
                sd = next((h_ for h_ in (self.side_of(base, p0), "right" if self.side_of(base, p0) == "left" else "left")
                           if self.plan_pick(n, h_) is not None), None)
                if sd is None or not self.pick(n, sd):
                    continue
                done = False
                for pl in sorted(plates, key=lambda q_: float(np.hypot(*(self.obj(q_)[0][:2] - p0[:2])))):
                    if pl not in self.table_items(T):
                        continue
                    tgt, clear = self.plate_spot(pl, self.kind(n))
                    if self.kind(n) == "glass" and clear < 0.06:
                        continue
                    if self.place(sd, tgt, yaws=None, drop=0.004, above=0.06, allow=(pl,) + tuple(self.riders(pl))):
                        done = True
                        moved += 1
                        break
                if not done and self.held[sd] == n:
                    # (no plate takes it from here: back where it was)
                    self.place(sd, p0 + np.array([0.0, 0.0, 0.002]), yaws=[yaw_of(R0)], drop=0.004, above=0.06)
            self.posture(0.79, 0.0)
            # (back from the table before the arms come down: hands just over the top would sweep its edge)
            self.back_off()
            self.home()
        self.log("[ref] consolidate %s: %d things onto the plates" % (T, moved))


    def pick(self, name, side, h=None, pitch=None):
        self.stats["picks"] += 1
        if self.held[side] and self.held[side] != name:
            # (a busy hand never closes on something else: opening it again would drop what it holds)
            self.log("[ref] pick %s (%s): the hand holds %s" % (name, side, self.held[side]))
            return False
        plan = self.plan_pick(name, side, h, pitch)
        if plan is None:
            self.stats["pick_fail"] += 1
            self.log("[ref] pick %s (%s): no grasp" % (name, side))
            return False
        poses, qs = plan
        if not self.transit(side, poses[0], allow=(name,) + tuple(self.riders(name))):
            self.stats["pick_fail"] += 1
            self.log("[ref] pick %s (%s): no transit" % (name, side))
            return False
        self.arm_to({side: poses[1]}, timeout=3.0, tol=0.004)
        got = self.hand(side, True)
        if got != name:
            self.log("[ref] pick %s (%s): the hand took %s" % (name, side, got))
            self.hand(side, False)
            self.arm_to({side: poses[0]}, timeout=4.0, tol=0.02)
            self.stats["pick_fail"] += 1
            return False
        allow = (name,) + tuple(self.riders(name))
        # (a tool held at one end, the cone by its bar, the mop by its handle, swings wide at the slightest turn
        # of the hand: straight up with the hand's attitude kept)
        lift = None if self.kind(name) in ("cone", "mop") else \
            self.joint_path(side, self.d.qpos[self.r.arm_q[side]].copy(), qs[2], allow=allow, step=0.04)
        if lift is not None:
            self.run_joints(side, lift, tol=0.02)
        else:
            ps, Rs = self.site(side)
            up = ps + (poses[2][0] - poses[1][0])
            self.lower(side, up, Rs, touch_N=25.0, allow=allow)
        return True

    def joint_path(self, side, q0, q1, allow=(), step=0.05, other_q=None, base=None, h=None, pitch=None):
        """The straight joint-space path q0 -> q1 in steps of at most `step` rad, each configuration checked (the
        robot's links and what the hands hold against everything else): the waypoints, or None if blocked."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = int(max(1, math.ceil(float(np.max(np.abs(q1 - q0))) / step)))
        out = []
        for k in range(1, n + 1):
            q = q0 + (q1 - q0) * (k / n)
            # (plans are made on the measured state: a few millimetres of graze at most; a link sunk deeper into
            # the cart or a table presses on it with a hundred newtons and more)
            if self.check_pose(side, q, allow=allow, other_q=other_q, tol=0.003, base=base, h=h, pitch=pitch):
                return None
            out.append(q)
        return out

    def run_joints(self, side, qs, tol=0.04, per_wp=60):
        """Stream joint waypoints (each held until the arm is near it): the executed path follows the planned one."""
        for k, q in enumerate(qs):
            self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
            last = k == len(qs) - 1
            for _ in range(per_wp if not last else per_wp * 3):
                self.step(2)
                if float(np.max(np.abs(self.d.qpos[self.r.arm_q[side]] - q))) < (0.012 if last else tol):
                    break
        return True

    def move_pose(self, side, p, R, allow=(), vias=True):
        """Move the hand to (p, R) along a checked joint-space path: straight from where the arm is, else through
        the raised ready postures."""
        q1, ep, er = self.ik(side, p, R)
        if ep > 0.008 or (R is not None and er > 0.06):
            return False
        if self.check_pose(side, q1, allow=allow):
            return False
        q0 = self.d.qpos[self.r.arm_q[side]].copy()
        path = self.joint_path(side, q0, q1, allow=allow)
        if path is None and vias:
            for via in (READY_HIGH[side], READY[side], TUCK[side]):
                a = self.joint_path(side, q0, via, allow=allow)
                b = self.joint_path(side, via, q1, allow=allow) if a is not None else None
                if a is not None and b is not None:
                    path = a + b
                    break
        if path is None:
            self.log("[ref] move_pose %s: no free joint path to %s" % (side, np.round(p, 3)))
            return False
        self.run_joints(side, path)
        return True

    def pick_checked(self, name, side):
        """pick() with every arm move checked: to the pre-grasp along a checked joint path, down to the grasp and up
        again along checked straight lines (where a thing lies among the cart's rims an unchecked approach can press an
        elbow onto the cart)."""
        if self.held[side]:
            return False
        plan = self.plan_pick(name, side)
        if plan is None:
            return False
        poses, qs = plan
        allow = (name,) + tuple(self.riders(name))
        if not self.transit(side, poses[0], allow=allow):
            return False
        if not self.lower(side, poses[1][0], poses[1][1], allow=allow):
            self.lower(side, poses[0][0], poses[0][1], allow=allow)
            return False
        if self.hand(side, True) != name:
            self.hand(side, False)
            self.lower(side, poses[0][0], poses[0][1], allow=allow)
            return False
        self.lower(side, poses[2][0], poses[2][1], allow=allow, touch_N=25.0)
        return True

    def transit(self, side, target, allow=()):
        """To a pre-grasp / pre-place pose along a checked joint-space path."""
        return self.move_pose(side, target[0], target[1], allow=allow)

    def carry_pose(self, side, z=1.00):
        """A carry pose for what the hand holds: in front of the chest on its side, the object's attitude kept."""
        b = self.base()
        ps, Rs = self.site(side)
        sg = 1.0 if side == "left" else -1.0
        name = self.held[side]
        kind = self.kind(name) if name and name != "cart" else None
        best = None
        for fwd, lat in ((0.30, 0.20), (0.26, 0.24), (0.34, 0.16), (0.24, 0.28)):
            p = np.array([b[0], b[1], 0.0]) + Rz(b[2]) @ np.array([fwd, sg * lat, z])
            for dyaw in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2):
                R = Rz(dyaw) @ Rs
                if kind in ("plate", "glass"):
                    # the approach horizontal, pointing forwards-inwards
                    ap = Rz(b[2] - sg * 0.5 + dyaw) @ np.array([1.0, 0.0, 0.0])
                    R = frame(ap, Rs[:, 1] if abs(Rs[2, 1]) > 0.7 else np.cross([0, 0, 1.0], ap) * sg)
                q, ep, er = self.ik(side, p, R)
                if ep < 0.006 and er < 0.05 and not self.check_pose(side, q):
                    return p, R
                if best is None or ep < best[2]:
                    best = (p, R, ep)
        return (best[0], best[1]) if best else None

    def hold_aside(self, side):
        """Lift what the `side` hand holds up and out to its side (clear of where the other hand works)."""
        b = self.base()
        sg = 1.0 if side == "left" else -1.0
        ps, Rs = self.site(side)
        for fwd, lat, z in ((0.10, 0.42, 1.22), (0.18, 0.38, 1.25), (0.05, 0.45, 1.15), (0.25, 0.35, 1.30)):
            p = np.array([b[0], b[1], 0.0]) + Rz(b[2]) @ np.array([fwd, sg * lat, z])
            for dyaw in (0.0, 0.5 * sg, 1.0 * sg, -0.5 * sg):
                R = Rz(dyaw) @ Rs
                q, ep, er = self.ik(side, p, R)
                if ep < 0.006 and er < 0.05 and not self.check_pose(side, q):
                    self.arm_to({side: (p, R)}, timeout=6.0, tol=0.02, frame="base")
                    return True
        return False

    def carry(self, sides=None, z=1.00):
        sides = [s for s in (sides or ("left", "right")) if self.held[s] and self.held[s] != "cart"]
        tg = {}
        for s in sides:
            cp = self.carry_pose(s, z)
            if cp is not None:
                tg[s] = cp
        if tg:
            self.arm_to(tg, timeout=8.0, tol=0.02, frame="base")

    def place(self, side, p_obj, yaws=None, drop=0.004, above=0.08, allow=(), release=True, clear_z=None):
        """Put the held object at p_obj (its frame's origin), upright, turned by one of `yaws` about the vertical:
        above it first (high enough that the object clears `clear_z`, a rim it passes over), down to `drop` over it,
        open the hand, up again."""
        self.stats["places"] += 1
        name = self.held[side]
        if not name or name not in self.rel or self.c.grips.holding(side) != name:
            self.sync_held()
            self.log("[ref] place: the %s hand holds nothing" % side)
            return False
        rel_p, rel_R = self.rel[name]
        yaws = list(yaws) if yaws is not None else list(np.linspace(-math.pi, math.pi, 12, endpoint=False))
        # turning the hand swings what it holds round the grasp point: the object's present heading first (a move
        # without a turn), then the nearest others
        yaw_now = yaw_of(self.obj(name)[1])
        if self.kind(name) in ("plate", "glass", "napkin"):
            yaws = [yaw_now] + yaws
        yaws.sort(key=lambda y: abs(wrap(y - yaw_now)))
        low = self.lowest(name)                      # its lowest point below its frame's origin (m)
        for yw in yaws:
            Ro = Rz(yw)
            Rs = Ro @ rel_R.T
            ps = np.asarray(p_obj, float) + np.array([0.0, 0.0, drop]) - Rs @ rel_p
            dz = above if clear_z is None else max(above, clear_z + 0.03 - (float(p_obj[2]) + drop - low))
            hi = ps + np.array([0.0, 0.0, dz])
            q1, e1, r1 = self.ik(side, hi, Rs)
            if e1 > 0.006 or r1 > 0.05 or self.check_pose(side, q1):
                continue
            q2, e2, r2 = self.ik(side, ps, Rs, seed=q1)
            if e2 > 0.006 or r2 > 0.05:
                continue
            bad = self.check_pose(side, q2, allow=allow)
            if bad:
                continue
            # to the high pose by a checked joint path (no unchecked move: else the next heading), then down
            if not self.move_pose(side, hi, Rs, allow=allow):
                continue
            down = self.joint_path(side, self.d.qpos[self.r.arm_q[side]].copy(), q2, allow=allow, step=0.04)
            if down is not None:
                self.run_joints(side, down, tol=0.02)
            else:
                self.lower(side, ps, Rs, allow=allow)
            if release:
                self.hand(side, False)
                self.arm_to({side: (hi, Rs)}, timeout=4.0, tol=0.02, frame="base")
            return True
        self.stats["place_fail"] += 1
        self.log("[ref] place %s (%s) at %s: no pose" % (name, side, np.round(p_obj, 3)))
        return False

    def lower(self, side, p, R, step=0.01, touch_N=8.0, allow=()):
        """Straight down (or along) to the grasp-site pose (p, R): joint targets streamed from a chain of IK solutions
        on the robot's measured state, 1 cm apart (the arm never leaves its branch, the hand keeps the line), stopping
        when what the hand holds touches down (its grip's load past touch_N). True if it got there."""
        cd = self.chk.d
        cd.qpos[:] = self.d.qpos
        p0, _ = self.site(side)
        p = np.asarray(p, float)
        n = max(1, int(math.ceil(float(np.linalg.norm(p - p0)) / step)))
        qk = self.d.qpos[self.r.arm_q[side]].copy()
        qt = quat_of(R)
        load0 = float(self.c.grips.state[side]["load"])
        for k in range(1, n + 1):
            pk = p0 + (p - p0) * (k / n)
            q, ep, er = self.nik.solve(cd.qpos, side, pk, qt, qk, iters=40, w_rot=0.35, natural=True, null_total=0.01)
            if ep > 0.006 or float(np.max(np.abs(q - qk))) > 0.25:
                return False
            if self.check_pose(side, q, allow=allow):
                return False
            cd.qpos[:] = self.d.qpos               # (check_pose used the scratch state)
            qk = q
            self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
            for _ in range(30):
                self.step(1)
                if self.held[side] and float(self.c.grips.state[side]["load"]) > load0 + touch_N:
                    return False
                if float(np.max(np.abs(self.d.qpos[self.r.arm_q[side]] - q))) < 0.008:
                    break
        return True

    def lift(self, side, dz, allow=(), step=0.01):
        """Straight up by dz like lower(), but the hand's attitude may give (down to a light attitude weight, up to 25
        degrees off) where the fixed-attitude line leaves the arm's reach: what the hand holds comes up all the same.
        True if it got there."""
        cd = self.chk.d
        cd.qpos[:] = self.d.qpos
        p0, R0 = self.site(side)
        qk = self.d.qpos[self.r.arm_q[side]].copy()
        qt = quat_of(R0)
        for k in range(1, int(math.ceil(dz / step)) + 1):
            pk = p0 + np.array([0.0, 0.0, min(dz, step * k)])
            q = None
            for w in (0.35, 0.15, 0.05):
                q_, ep, er = self.nik.solve(cd.qpos, side, pk, qt, qk, iters=40, w_rot=w, natural=True,
                                            null_total=0.01)
                if ep <= 0.006 and er < 0.45 and float(np.max(np.abs(q_ - qk))) <= 0.25:
                    q = q_
                    break
            if q is None or self.check_pose(side, q, allow=allow):
                return False
            cd.qpos[:] = self.d.qpos
            qk = q
            self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
            for _ in range(30):
                self.step(1)
                if float(np.max(np.abs(self.d.qpos[self.r.arm_q[side]] - q))) < 0.008:
                    break
        return True

    def low_point(self, name):
        """The lowest point (room z) of a roll-up as it lies now: its box collider (its length along its frame's x,
        its frame's origin at its underside) at the attitude it has."""
        po, Ro = self.obj(name)
        L, r = O.ROLLUP["len"], O.ROLLUP["r"]
        cz = float(po[2] + Ro[2, 2] * 0.88 * r)
        return cz - 0.5 * L * abs(float(Ro[2, 0])) - 0.95 * r * abs(float(Ro[2, 1])) - 0.88 * r * abs(float(Ro[2, 2]))

    def lowest(self, name):
        """How far below its frame's origin an object reaches (upright)."""
        kind = self.kind(name)
        return {"plate": 0.0, "glass": 0.0, "fork": 0.0, "knife": 0.0, "napkin": 0.0, "rollup": 0.0,
                "cloth": O.CLOTH["size"][2] / 2}.get(kind, 0.0)

    # ------------------------------------------------------------------------------------------ the cart
    CART_BEHIND = 0.40           # m: the pelvis this far behind the handle while pushing

    def cart_pose(self):
        return self.c.cart.pose()

    def cart_to_world(self, p_local, pose=None):
        x, y, yaw = self.cart_pose() if pose is None else pose
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([x + c * p_local[0] - s * p_local[1], y + s * p_local[0] + c * p_local[1]] +
                        ([p_local[2]] if len(p_local) > 2 else []))

    def push_spot(self, pose=None):
        """Where the robot stands to push the cart at `pose` (behind its handle, facing along the cart)."""
        x, y, yaw = self.cart_pose() if pose is None else pose
        k = RM.CART["handle_x"] - self.CART_BEHIND
        return np.array([x + math.cos(yaw) * k, y + math.sin(yaw) * k, yaw])

    def cart_grip(self):
        if self.c.cart.grasp:
            return True
        if any(self.held[sd] and self.held[sd] != "cart" for sd in ("left", "right")):
            # (the hands must be free for the handle: what they still hold is put away first, never dropped)
            if not self.free_hands():
                self.log("[ref] cart_grip: the hands are not free (%s)" % (dict(self.held),))
                return False
        sp = self.push_spot()
        if not self.approach(sp[0], sp[1], sp[2], back=0.30):
            return False
        self.posture(0.79, 0.0)
        x, y, yaw = self.cart_pose()
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        tg = {}
        for side, dy in (("left", 0.12), ("right", -0.12)):
            p = self.cart_to_world((RM.CART["handle_x"], dy, RM.CART["handle_z"]))
            tg[side] = (p, frame(fwd, (0, 0, 1.0) if side == "left" else (0, 0, -1.0)))
        self.arm_to(tg, timeout=8.0, tol=0.005)
        self.hand("left", True)
        self.hand("right", True)
        if not (self.held["left"] == "cart" and self.held["right"] == "cart"):
            self.log("[ref] cart_grip: the handle was not taken (%s)" % (self.held,))
            self.hand("left", False)
            self.hand("right", False)
            self.home()
            return False
        tgb = {s: self.site(s) for s in ("left", "right")}
        self.arm_to(tgb, timeout=1.0, tol=0.01, frame="base")
        return True

    def cart_release(self):
        if not self.c.cart.grasp:
            return
        self.hand("left", False)
        self.hand("right", False)
        b = self.base()
        self.goto(b[0] - 0.12 * math.cos(b[2]), b[1] - 0.12 * math.sin(b[2]), b[2], tol=0.03)
        self.home()

    def cart_go(self, x, y, tol=0.02, timeout=60.0):
        """Move the held cart's centre to (x, y), its heading kept: a straight move of the robot behind it."""
        q = self.cart_pose()
        sp = self.push_spot((x, y, q[2]))
        self.c.command({"base_goal": [float(sp[0]), float(sp[1]), float(sp[2])]})
        for _ in range(int(timeout / 0.05)):
            self.step(5)
            if not self.c.cart.grasp:
                self.log("[ref] cart_go: the handle slipped at %s" % (np.round(self.cart_pose(), 3),))
                self.c.command({"base_velocity": [0.0, 0.0, 0.0]})
                self.step(20)
                return False
            q = self.cart_pose()
            if math.hypot(q[0] - x, q[1] - y) < tol:
                break
        self.step(10)
        q = self.cart_pose()
        return math.hypot(q[0] - x, q[1] - y) < tol * 2

    def cart_route(self, x, y):
        """Push the cart to (x, y) (its heading kept): sideways in the east zone to the target row's line, then
        straight along it (an aisle), or the reverse when it starts in an aisle."""
        q = self.cart_pose()
        if not self.cart_grip():
            return False
        EAST = 6.75
        pts = []
        if q[0] < EAST - 0.05:                 # in an aisle: straight out to the east zone first
            pts.append((EAST, q[1]))
        if abs(y - q[1]) > 0.02:
            pts.append((max(EAST, x if x > EAST else EAST), y))
        pts.append((x, y))
        for px, py in pts:
            if not self.cart_go(px, py):
                return False
        return True

    # ------------------------------------------------------------------------------------------ the tub
    TUB_STACKS = ((0.145, 0.0), (-0.145, 0.0))             # offsets from the tub's middle (cart frame)
    TUB_GLASSES = ((0.24, 0.128), (0.24, -0.128), (-0.24, 0.128), (-0.24, -0.128))

    def tub_rim(self):
        return float(self.c.meta["cart"]["tub_inside"][5])

    def tub_mid(self):
        x0, y0, z0, x1, y1, z1 = self.c.meta["cart"]["tub_inside"]
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2, z0])

    def in_tub(self, name):
        po, _ = self.obj(name)
        loc = self.c.cart._to_cart(po)
        x0, y0, z0, x1, y1, z1 = self.c.meta["cart"]["tub_inside"]
        return x0 - 0.02 < loc[0] < x1 + 0.02 and y0 - 0.02 < loc[1] < y1 + 0.02 and z0 - 0.03 < loc[2] < z1 + 0.15

    def tub_contents(self):
        return [n for n in self.c.layout["loose"] if n not in self.c.shift.parked and n not in self.held.values()
                and self.in_tub(n)]

    def stack_top(self, k):
        """The top (room z) of plate stack k and the number of plates in it."""
        mid = self.tub_mid()
        cxl, cyl = mid[0] + self.TUB_STACKS[k][0], mid[1] + self.TUB_STACKS[k][1]
        top, n = mid[2], 0
        for name in self.tub_contents():
            po, Ro = self.obj(name)
            loc = self.c.cart._to_cart(po)
            if math.hypot(loc[0] - cxl, loc[1] - cyl) < O.PLATE["r"] + 0.02:
                kind = self.kind(name)
                h = {"plate": O.PLATE["h"], "glass": O.GLASS["h"], "napkin": 0.035, "rollup": 0.04}.get(kind, 0.008)
                top = max(top, po[2] + h)
                n += kind == "plate"
        return top, n

    def tub_target(self, name):
        """Where the next `name` goes in the tub: (room position of its frame's origin, yaw options)."""
        kind = self.kind(name)
        mid = self.tub_mid()
        yaw_c = self.cart_pose()[2]
        if kind == "plate":
            # (not onto a stack with a napkin or cutlery lying on it while the other has none: a plate let go onto a
            # light napkin squeezes it out at speed)
            tops = [self.stack_top(k) for k in range(2)]
            smalls = self._stack_smalls()
            k = min(range(2), key=lambda i: (smalls[i] > 0, tops[i][1], i))
            off = self.TUB_STACKS[k]
            p = self.cart_to_world((mid[0] + off[0], mid[1] + off[1], tops[k][0] + 0.002))
            return p, None
        if kind == "glass":
            used = []
            for n in self.tub_contents():
                if self.kind(n) == "glass":
                    used.append(self.c.cart._to_cart(self.obj(n)[0])[:2])
            for off in self.TUB_GLASSES:
                loc = np.array([mid[0] + off[0], mid[1] + off[1]])
                if all(np.linalg.norm(loc - u) > 0.06 for u in used):
                    return self.cart_to_world((loc[0], loc[1], mid[2] + 0.002)), None
            return None, None
        # small things: on the stack that already has small things on it, else the one with more plates (the next
        # plate goes to the other one), else stack 1 (the first plate goes to stack 0)
        tops = [self.stack_top(k) for k in range(2)]
        smalls = self._stack_smalls()
        k = max(range(2), key=lambda i: (smalls[i], tops[i][1], i))
        off = self.TUB_STACKS[k]
        # (a deterministic spread by name: Python's str hash changes from process to process)
        hv = sum((i + 1) * ord(ch) for i, ch in enumerate(name))
        jit = (0.03 * ((hv % 5) - 2), 0.025 * ((hv // 5 % 3) - 1))
        p = self.cart_to_world((mid[0] + off[0] + jit[0], mid[1] + off[1] + jit[1], tops[k][0] + 0.03))
        return p, yaw_c + np.array([0.0, math.pi / 2, math.pi, -math.pi / 2])

    def _stack_smalls(self):
        """How many small things (napkins, cutlery) lie in the tub over each plate stack."""
        mid = self.tub_mid()
        out = [0, 0]
        for n in self.tub_contents():
            if self.kind(n) not in ("napkin", "fork", "knife"):
                continue
            loc = self.c.cart._to_cart(self.obj(n)[0])
            for k, off in enumerate(self.TUB_STACKS):
                if math.hypot(loc[0] - (mid[0] + off[0]), loc[1] - (mid[1] + off[1])) < O.PLATE["r"]:
                    out[k] += 1
        return out

    def drop_target(self, name):
        """Where to let `name` go over the tub: its slot (tub_target) at just above the rim, so the hand and what it
        holds stay over the walls (a plate drops at most ~13 cm onto its stack: well under a dish's breaking speed)."""
        p, yaws = self.tub_target(name)
        if p is None:
            mid = self.tub_mid()
            p, yaws = self.cart_to_world((mid[0], mid[1], mid[2])), None
        p = np.array(p, float)
        p[2] = max(float(p[2]), self.tub_rim() + 0.015 + self.lowest(name))
        return p, yaws

    # ------------------------------------------------------------------------------------------ stances
    def stance_for(self, side, poses, bases, heights=(0.79,), pitches=(0.0, 0.25, 0.45), held=(), allow=()):
        """The first (base, h, pitch) from which the `side` hand reaches every (p, R) of `poses` (each solved from the
        previous solution) without a collision; None if none does."""
        for base in bases:
            for h in heights:
                for pitch in pitches:
                    seed = None
                    ok = True
                    for p, R in poses:
                        q, ep, er = self.ik(side, p, R, base=np.asarray(base), h=h, pitch=pitch, seed=seed)
                        if ep > 0.006 or (R is not None and er > 0.05):
                            ok = False
                            break
                        if self.check_pose(side, q, held=held, allow=allow, base=np.asarray(base), h=h, pitch=pitch):
                            ok = False
                            break
                        seed = q
                    if ok:
                        return np.asarray(base, float), h, pitch
        return None

    # ------------------------------------------------------------------------------------------ tables
    def table_items(self, T):
        """The guests' things still on table T (not held, not in the tub, not delivered)."""
        t = RM.tables()[T]
        (cx, cy), (sx, sy) = t["center"], t["size"]
        ids = {st["id"] for st in t["seats"]}
        out = []
        for n, (kind, _p) in self.c.layout["loose"].items():
            if kind not in O.GUEST_KINDS or n.split("_", 1)[1] not in ids:
                continue
            if n in self.c.shift.parked or n in self.held.values():
                continue
            po, _ = self.obj(n)
            if abs(po[0] - cx) < sx / 2 + 0.05 and abs(po[1] - cy) < sy / 2 + 0.05 and po[2] > RM.TABLE_TOP_Z - 0.05:
                out.append(n)
        return out

    def end_spots(self, T, gap=WORK_GAP):
        t = RM.tables()[T]
        (cx, cy), (sx, sy) = t["center"], t["size"]
        return [(cx - sx / 2 - gap, cy, 0.0), (cx + sx / 2 + gap, cy, math.pi)]

    def cart_park(self, T):
        """Where the cart stands while table T is worked: in the aisle beside the table's row, the tub level with
        the table."""
        t = RM.tables()[T]
        (cx, cy) = t["center"]
        tx = RM.CART["tub_at"][0]
        # (0.18 m off the chair backs: the robot behind the push handle stands clear of them; the aisle's other side
        # stays passable)
        if t["row"] == "S":
            y = self.AISLE1[1] - 0.18 - RM.CART["size"][1] / 2
        elif t["row"] == "M":
            y = self.AISLE1[0] + 0.18 + RM.CART["size"][1] / 2
        else:
            y = self.AISLE2[0] + 0.18 + RM.CART["size"][1] / 2
        return (cx + tx, y)

    AISLE1 = (1.65, 3.27)          # y range between the chair backs of rows S and M
    AISLE2 = (4.92, 6.54)

    def load_spots(self):
        """Where the robot stands to load the tub: beside the parked cart's tub, facing it (the side away from the
        aisle's near chairs first)."""
        x, y, yaw = self.cart_pose()
        mid = self.tub_mid()
        tw = self.cart_to_world((mid[0], mid[1]))
        half = RM.CART["size"][1] / 2
        out = []
        for gap in (0.18, 0.22, 0.15):
            for sg in (-1.0, 1.0):
                py = tw[1] + sg * (half + gap)
                out.append((tw[0], py, math.atan2(-sg, 0.0)))
        aisle = self.AISLE1 if y < 4.0 else self.AISLE2
        out.sort(key=lambda b: 0 if aisle[0] + 0.30 < b[1] < aisle[1] - 0.30 else 1)
        return out

    def side_of(self, base, p):
        """'left' or 'right' of the robot at `base` for a point p."""
        h = np.array([math.cos(base[2]), math.sin(base[2])])
        d = np.asarray(p[:2]) - np.asarray(base[:2])
        return "left" if h[0] * d[1] - h[1] * d[0] > 0 else "right"

    PRIORITY = {"plate": 0, "glass": 1, "napkin": 2, "fork": 3, "knife": 3}

    def bus(self, T, deadline=None):
        """Clear table T into the tub (two things a trip, a plate with what was stacked on it counting as one)."""
        if T not in self._consolidated:
            self._consolidated.add(T)
            self.consolidate(T, deadline=deadline)
        park = self.cart_park(T)
        q = self.cart_pose()
        if math.hypot(q[0] - park[0], q[1] - park[1]) > 0.06:
            if not self.cart_route(*park):
                self.log("[ref] bus %s: the cart did not get to its spot" % T)
            self.cart_release()
        dead_spots = set()
        for rnd in range(40):
            if self.out_of_time(60.0) or (deadline is not None and self.t > deadline):
                break
            items = sorted(self.table_items(T), key=lambda n: (self.PRIORITY[self.kind(n)], n))
            if not items:
                return True
            # the end spot (the working stance, else the close one that reaches further in) and the pair of things
            # (one per hand) to take from it
            best = None
            stances = [(sp, WORK_H, WORK_PITCH) for sp in self.end_spots(T)] + \
                [(sp, self.WIPE_H, self.WIPE_PITCH) for sp in self.end_spots(T, gap=self.WIPE_GAP)]
            for k_st, (spot, h_, p_) in enumerate(stances):
                if tuple(np.round(spot, 3)) in dead_spots:
                    continue
                if best is not None and k_st >= 2:
                    break                        # (the close stance only when the working one reaches nothing)
                base = np.array(spot)
                pair = {}
                # (a plan for a stance the robot is not at is made on the modelled posture, a centimetre or so off the
                # real one: an item counts as reachable only if it also is from 2 cm further back)
                back2 = base - np.array([0.02 * math.cos(base[2]), 0.02 * math.sin(base[2]), 0.0])
                for n in items:
                    side = self.side_of(base, self.obj(n)[0])
                    for sd in (side, "right" if side == "left" else "left"):
                        if sd in pair:
                            continue
                        if self.plan_pick(n, sd, h=h_, pitch=p_, base=base) is not None and \
                                self.plan_pick(n, sd, h=h_, pitch=p_, base=back2) is not None:
                            pair[sd] = n
                            break
                    if len(pair) == 2:
                        break
                score = sum(1.0 / (1 + self.PRIORITY[self.kind(n)]) for n in pair.values())
                if pair and (best is None or score > best[0]):
                    best = (score, spot, pair, h_, p_)
            if best is None:
                self.log("[ref] bus %s: nothing reachable of %s" % (T, items))
                return False
            _, spot, pair, h_, p_ = best
            if not self.approach(*spot, h=h_, pitch=p_):
                self.log("[ref] bus %s: could not walk to %s" % (T, np.round(spot, 2)))
                return False
            # at the stance itself (the arm's way from where it is now counts): the planned thing first, else the
            # next one this hand can take
            got, taken = [], set()
            for sd in sorted(pair):
                for n in [pair[sd]] + [m for m in items if m not in pair.values()]:
                    if n in taken or self.held[sd]:
                        continue
                    if self.plan_pick(n, sd) is None:
                        continue
                    if self.pick(n, sd):
                        got.append(sd)
                        taken.add(n)
                    break
            self.posture(0.79, 0.0)
            if not got:
                dead_spots.add(tuple(np.round(spot, 3)))
                self.back_off()
                self.home()
                if len(dead_spots) >= 4:
                    self.log("[ref] bus %s: nothing more to take of %s" % (T, items))
                    return False
                continue
            dead_spots.clear()
            self.carry(z=1.02)
            self.back_off()
            self.load(got)
        return not self.table_items(T)

    def load(self, sides):
        """Put what the hands hold into the tub, from a stance beside it (walkable, the one where every held thing
        has a pose over its slot first; the next one when a stance does not get it all in)."""
        sides = [s for s in sides if self.held[s] and self.held[s] != "cart"]
        if not sides:
            self.home()
            return
        others = list(floor_boxes(extra=self.obstacles(cart=False)))

        def walkable(p, r=0.24):
            for (x0, y0, x1, y1) in others:
                if math.hypot(max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1)) < r:
                    return False
            return True

        def n_posed(spot):
            n = 0
            for s in sides:
                p, yaws = self.drop_target(self.held[s])
                if p is None:
                    n += 1
                    continue
                rel_p, rel_R = self.rel[self.held[s]]
                yaw_now = yaw_of(self.obj(self.held[s])[1])
                cand = [yaw_now] + list(yaws if yaws is not None else np.linspace(-math.pi, math.pi, 8, endpoint=False))
                cand.sort(key=lambda y: abs(wrap(y - yaw_now)))
                for yw in cand:
                    Rs = Rz(yw) @ rel_R.T
                    ps = np.asarray(p) - Rs @ rel_p
                    hi = ps + np.array([0, 0, 0.06])
                    if self.stance_for(s, [(hi, Rs), (ps, Rs)], [spot],
                                       heights=(0.79,), pitches=(0.15, 0.30), allow=("cart",)) is not None:
                        n += 1
                        break
            return n

        # (the stance and the point 0.30 m behind it, where approach() walks in from)
        spots = [sp for sp in self.load_spots() if walkable(sp) and
                 walkable((sp[0] - 0.30 * math.cos(sp[2]), sp[1] - 0.30 * math.sin(sp[2])))]
        if not spots:
            self.log("[ref] load: no walkable stance by the tub")
            spots = self.load_spots()[:2]
        scored = []
        for k, sp in enumerate(spots):
            n = n_posed(sp)
            scored.append((-n, k, sp))
            if n == len(sides):
                break
        scored.sort()
        order = [sp for _, _, sp in scored] + [sp for sp in spots if all(sp is not t for _, _, t in scored)]
        for st in order[:3]:
            left = [s for s in sides if self.held[s] and self.held[s] != "cart"]
            if not left:
                break
            if not self.approach(*st, back=0.30, h=0.79, pitch=0.25, high=True):
                self.log("[ref] load: the stance %s was not reached" % (np.round(st, 2),))
                continue
            for s in left:
                name = self.held[s]
                other = "right" if s == "left" else "left"
                if self.held[other] and self.held[other] != "cart":
                    self.hold_aside(other)
                p, yaws = self.drop_target(name)
                if not self.place(s, p, yaws=yaws, drop=0.0, above=0.06):
                    # (no pose over its slot: over the tub's middle)
                    mid = self.tub_mid()
                    pm = np.array(self.cart_to_world((mid[0], mid[1], self.tub_rim() + 0.02 + self.lowest(name))),
                                  float)
                    if not self.place(s, pm, drop=0.0, above=0.06):
                        self.log("[ref] load: %s stays in the %s hand" % (name, s))
            self.posture(0.79, 0.0)
            # (away from the tub before the arms come down: a hand over its rim would sweep the wall)
            self.back_off(0.25)
            if any(self.held[s] and self.held[s] != "cart" for s in sides):
                self.carry()
        self.home()

    # ------------------------------------------------------------------------------------------ the cloth
    def cart_side_spots(self, local_xy, gaps=(0.17, 0.21, 0.14)):
        """Stances beside the parked cart level with a point of its top (cart frame), facing the cart; those inside
        the aisle first."""
        x, y, yaw = self.cart_pose()
        w = self.cart_to_world(local_xy)
        half = RM.CART["size"][1] / 2
        out = []
        for gap in gaps:
            for sg in (-1.0, 1.0):
                out.append((w[0], w[1] + sg * (half + gap), math.atan2(-sg, 0.0)))
        aisle = self.AISLE1 if y < 4.0 else self.AISLE2
        out.sort(key=lambda b: 0 if aisle[0] + 0.30 < b[1] < aisle[1] - 0.30 else 1)
        return out

    def bucket_rim(self):
        return RM.CART["top_z"] + RM.BUCKET["h"]

    def cloth_side(self, spot):
        """The hand on the side away from the tub at a stance beside the bucket."""
        tw = self.cart_to_world(tuple(self.tub_mid()[:2]))
        return "left" if self.side_of(np.asarray(spot), tw) == "right" else "right"

    def cloth_take(self, side=None):
        """Take the cloth out of the sanitizer bucket (with the hand away from the tub); returns the hand or None."""
        for sd in ("left", "right"):
            if self.held[sd] == "cloth":
                return sd
        # (the usual stance first; a lower or deeper bow, or the other hand, where the cloth lies awkwardly)
        tries = [(spot, self.cloth_side(spot), 0.79, 0.30) for spot in self.cart_side_spots(RM.CART["bucket_at"])]
        for (h_, pt_) in ((0.79, 0.45), (0.79, 0.15), (0.74, 0.52)):
            tries += [(spot, self.cloth_side(spot), h_, pt_) for spot in self.cart_side_spots(RM.CART["bucket_at"])]
        tries += [(spot, "left" if self.cloth_side(spot) == "right" else "right", 0.79, 0.30)
                  for spot in self.cart_side_spots(RM.CART["bucket_at"])]
        attempts = 0
        for spot, side, h_, pt_ in tries:
            base = np.array(spot)
            if attempts >= 3:
                break
            if self.plan_pick("cloth", side, h=h_, pitch=pt_, base=base) is None:
                continue
            attempts += 1
            self.approach(*spot, back=0.30, h=h_, pitch=pt_, high=True)
            ok = self.pick("cloth", side)
            if not ok and self.held[side] is None and self.c.process.in_bucket(self.obj("cloth")[0]):
                # (the hand closed on nothing: the next stance or hand)
                self.posture(0.79, 0.0)
                self.back_off(0.20)
                self.home()
                continue
            if ok:
                ps, Rs = self.site(side)
                self.arm_to({side: (ps + np.array([0, 0, max(0.0, self.bucket_rim() + 0.03 - ps[2])]), Rs)},
                            timeout=4.0, tol=0.01, frame="base")
            self.posture(0.79, 0.0)
            if ok:
                # (back from the cart before the hand comes to its carry pose: the cloth hangs over the rims; carried
                # over the height of the cart's push handle)
                self.back_off(0.25)
                self.carry([side], z=1.14)
            return side if ok and self.sync_held() and self.held[side] == "cloth" else None
        po, Ro = self.obj("cloth")
        self.log("[ref] cloth_take: no stance (the cloth at %s, its normal %s, in the bucket %s)" % (
            np.round(po, 3), np.round(Ro[:, 2], 2), self.c.process.in_bucket(po)))
        return None

    def cloth_back(self, side=None, leave=True, dip_s=1.4, _retry=0):
        """Back into the sanitizer bucket: the cloth held flat over the pail's middle, a few centimetres above its
        rim, and let go (it drops onto the rack, under the solution)."""
        side = next((sd for sd in ("left", "right") if self.held[sd] == "cloth"), None)
        if side is None:
            return False
        bx, by = RM.CART["bucket_at"]
        half_t = O.CLOTH["size"][2] / 2
        target = self.cart_to_world((bx, by, self.bucket_rim() + 0.035 + half_t))
        rel_p, rel_R = self.rel["cloth"]
        # (carried high to the cart: the cloth hangs under the hand; at 1.14 m it passes 7 cm over the cart's push
        # handle, which stands beside the bucket at the cloth's height when the hand is lower, and over the rims)
        self.carry([side], z=1.14)
        for spot in self.cart_side_spots((bx, by)):
            self.approach(*spot, back=0.30, h=0.79, pitch=0.30, high=True)
            yaw_now = yaw_of(self.obj("cloth")[1])
            done = False
            for dyaw in sorted(np.linspace(-math.pi, math.pi, 12, endpoint=False), key=lambda v: abs(v)):
                Rs = Rz(yaw_now + dyaw) @ rel_R.T
                ps = np.asarray(target, float) - Rs @ rel_p
                q, ep, er = self.ik(side, ps, Rs)
                if ep > 0.006 or er > 0.05 or self.check_pose(side, q, allow=("cart",)):
                    continue
                if not self.move_pose(side, ps, Rs, allow=("cart",)):
                    continue
                self.arm_to({side: (ps, Rs)}, timeout=3.0, tol=0.008)
                done = True
                break
            if not done:
                continue
            # (the cloth's middle where it really hangs, over the pail's middle before it is let go: it fits the pail
            # with a few centimetres to spare)
            for _ in range(2):
                pc_, _ = self.obj("cloth")
                err = np.array([target[0] - pc_[0], target[1] - pc_[1], 0.0])
                if float(np.linalg.norm(err)) < 0.015:
                    break
                ps2, Rs2 = self.site(side)
                if not self.move_pose(side, ps2 + err, Rs2, allow=("cart",), vias=False):
                    break
                self.arm_to({side: (ps2 + err, Rs2)}, timeout=2.0, tol=0.006)
            # (let go while the hand moves down: a cloth let go from a hand held still stays at rest where it was let
            # go, the way things at rest sleep in the cell until something touches them)
            ps4, Rs4 = self.site(side)
            pb4, Rb4 = self.to_base(ps4 - np.array([0.0, 0.0, 0.05]), Rs4)
            self.c.command({f"{side}_arm": {"pose": [float(v) for v in pb4] + [float(v) for v in quat_of(Rb4)],
                                            "frame": "base", "path": "line"}})
            for _ in range(40):
                self.step(1)
                if self.site(side)[0][2] < ps4[2] - 0.006:
                    break
            self.hand(side, False)
            self.sync_held()
            self.step(int(dip_s / TICK))
            ps, Rs = self.site(side)
            self.arm_to({side: (ps + np.array([0, 0, 0.08]), Rs)}, timeout=3.0, tol=0.02, frame="base")
            if not self.c.process.in_bucket(self.obj("cloth")[0]) and _retry < 2:
                # (it caught the rim: taken again from where it lies, and let go once more)
                self.log("[ref] cloth_back: the cloth lies at %s, not in the bucket: again" % (
                    np.round(self.obj("cloth")[0], 3),))
                if self.pick_checked("cloth", side):
                    return self.cloth_back(side, leave, dip_s, _retry=_retry + 1)
            self.posture(0.79, 0.0)
            self.back_off(0.25)
            self.home()
            return self.c.process.in_bucket(self.obj("cloth")[0])
        self.log("[ref] cloth_back: no pose over the bucket")
        return False

    def press_stroke(self, side, pts, yaw_obj, press=0.004, lift=0.05):
        """Slide the held cloth's centre along the line pts[0] -> pts[-1] (room points on the surface), its face
        pressed `press` into the top (the folded cloth gives: about 3 N), its long axis along `yaw_obj`: the stroke is
        trimmed to the part one arm branch follows continuously with the cloth flat (solved on the robot's measured
        state, from a good solution near the middle grown both ways), then streamed as joint targets. A force near
        the pad's end lets the rest of the stroke up 3 mm. False if the cloth slipped."""
        rel_p, rel_R = self.rel["cloth"]
        Rs = Rz(yaw_obj) @ rel_R.T
        qt = quat_of(Rs)
        half_t = O.CLOTH["size"][2] / 2

        def site_for(c, dz):
            return np.asarray(c, float) + np.array([0.0, 0.0, half_t - dz]) - Rs @ rel_p

        cd = self.chk.d
        cd.qpos[:] = self.d.qpos                 # the robot as it stands now
        P = np.asarray(pts, float)
        a, b = P[0], P[-1]
        n_u = max(5, int(round(float(np.linalg.norm(b - a)) / 0.008)) + 1)
        us = np.linspace(0.0, 1.0, n_u)

        def sol(u, dz, seed, drift):
            return self.nik.solve(cd.qpos, side, site_for(a + (b - a) * u, dz), qt, seed, iters=40, w_rot=0.35,
                                  natural=True, null_total=drift)
        seeds = [self.d.qpos[self.r.arm_q[side]].copy(), np.array(READY[side], float)] + \
            [np.array(v, float) for v in REACH_SEEDS[side]]
        mid = None
        for frac in (0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2):
            km = int(round(frac * (n_u - 1)))
            for sd in seeds:
                q, ep, er = sol(us[km], press, sd, None)
                if ep < 0.003 and er < 0.04 and (mid is None or ep + 0.3 * er < mid[2]):
                    mid = (km, q, ep + 0.3 * er)
            if mid is not None:
                break
        if mid is None:
            return True
        chain = {mid[0]: mid[1]}
        for direction in (-1, 1):
            k, qk = mid[0], mid[1]
            while 0 <= k + direction < len(us):
                q, ep, er = sol(us[k + direction], press, qk, 0.02)
                if not (ep < 0.003 and er < 0.035 and float(np.max(np.abs(q - qk))) < 0.10):
                    break
                k, qk = k + direction, q
                chain[k] = q
        ks = sorted(chain)
        if len(ks) < 3:
            return True
        if self.trace:
            self.log("    [stroke] %s -> %s at t %.2f" % (np.round((a + (b - a) * us[ks[0]])[:2], 3),
                                                        np.round((a + (b - a) * us[ks[-1]])[:2], 3), self.t))
        pr = self.c.process

        def normal():
            key, N = pr._contact_force(pr.cloth_geom, pr.surface_geoms)
            return N if key is not None else 0.0

        state = {"off": 0.0, "peak": 0.0, "lets": 0}

        def to(q, ticks=25, tol=0.010, watch=True, k=None):
            self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
            for _ in range(ticks):
                self.step(1)
                if watch:
                    N = normal()
                    state["peak"] = max(state["peak"], N)
                    if N > 5.5 and k is not None and state["lets"] < 2:
                        # (pressing towards the pad's end: the rest of the stroke 3 mm higher)
                        state["off"] += 0.003
                        state["lets"] += 1
                        q2, ep2, er2 = sol(us[k], press - state["off"], q, 0.01)
                        if ep2 < 0.006:
                            q = q2
                            self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
                if float(np.max(np.abs(self.d.qpos[self.r.arm_q[side]] - q))) < tol:
                    break
            return self.c.grips.holding(side) == "cloth"

        # over the start: from a lane's end next to it straight across (lifted), from further a checked joint path
        k0 = ks[0]
        q_up, ep_u, er_u = sol(us[k0], -lift, chain[k0], 0.02)
        ps0, _ = self.site(side)
        up = site_for(a + (b - a) * us[k0], -lift)
        if not (float(np.linalg.norm(ps0 - up)) < 0.30 and abs(ps0[2] - up[2]) < 0.025):
            path = self.joint_path(side, self.d.qpos[self.r.arm_q[side]].copy(), q_up, allow=())
            if path is not None:
                self.run_joints(side, path)
        else:
            q_now = self.d.qpos[self.r.arm_q[side]].copy()
            for w in np.linspace(0.25, 1.0, 4):
                to(q_now + (q_up - q_now) * w, ticks=40, tol=0.02, watch=False)
        to(q_up, ticks=60, tol=0.008, watch=False)
        # down
        for dz in np.linspace(-lift, press, 5)[1:]:
            q, ep, er = sol(us[k0], float(dz), chain[k0], 0.01)
            if not to(q, ticks=30, tol=0.008, k=None):
                break
        # along
        ok = True
        for k in ks:
            q = chain[k]
            if state["off"] > 0:
                q, ep, er = sol(us[k], press - state["off"], q, 0.01)
            if not to(q, k=k):
                ok = False
                break
            if self.trace and self.trace_ticks > 0:
                self.trace_ticks -= 1
                want = site_for(a + (b - a) * us[k], press - state["off"])
                ps_, Rs_ = self.site(side)
                po_, Ro_ = self.obj("cloth")
                cd.qpos[self.r.arm_q[side]] = self.d.qpos[self.r.arm_q[side]]
                mujoco.mj_kinematics(self.m, cd)
                fk = cd.site_xpos[self.r.grasp_site[side]].copy()
                cd.qpos[self.r.arm_q[side]] = q
                mujoco.mj_kinematics(self.m, cd)
                fkq = cd.site_xpos[self.r.grasp_site[side]].copy()
                rel_now = Rs_.T @ (po_ - ps_)
                self.log("        k %d N %.1f site-want %s actual %s fk(actual q) %s fk(cmd q) %s |q-cmd| %.3f cloth z %.4f (want %.4f) rel %s (grasp %s)" % (
                    k, normal(), np.round(want, 4), np.round(ps_, 4), np.round(fk, 4), np.round(fkq, 4),
                    float(np.max(np.abs(self.d.qpos[self.r.arm_q[side]] - q))), po_[2],
                    RM.TABLE_TOP_Z + half_t - (press - state["off"]), np.round(rel_now, 4), np.round(rel_p, 4)))
        if self.trace:
            self.log("      along: peak N %.1f lets %d" % (state["peak"], state["lets"]))
        if not ok:
            self.log("[ref] press_stroke: the cloth slipped")
            self.sync_held()
            return False
        # up
        kl = ks[-1]
        for dz in np.linspace(press - state["off"], -lift, 4)[1:]:
            q, ep, er = sol(us[kl], float(dz), self.d.qpos[self.r.arm_q[side]].copy(), 0.01)
            to(q, ticks=30, tol=0.012, watch=False)
        return True

    WIPE_GAP, WIPE_H, WIPE_PITCH = 0.10, 0.74, 0.52
    WIPE_BAND = 0.36               # m: a hand wipes lanes up to this far to its own side of the pelvis (and 0.06 across)

    def swap_cloth(self, T):
        """Hand the cloth to the other hand: laid flat on the top in front of the robot, taken up by the other hand."""
        side = next((sd for sd in ("left", "right") if self.held[sd] == "cloth"), None)
        if side is None:
            return None
        b = self.base()
        hd = np.array([math.cos(b[2]), math.sin(b[2])])
        zt = RM.TABLE_TOP_Z
        half_t = O.CLOTH["size"][2] / 2
        for u in (0.30, 0.26, 0.34):
            p = np.array([b[0] + hd[0] * u, b[1] + hd[1] * u, zt + half_t])
            if self.place(side, p, yaws=[b[2], b[2] + math.pi], drop=0.003, above=0.05, allow=(f"{T}_top",)):
                break
        else:
            return None
        other = "right" if side == "left" else "left"
        self.sync_held()
        # (the free hand up and out of the way: home, at this lean, would put its wrist on the top)
        self.ready([side], high=True)
        if self.pick("cloth", other):
            ps, Rs = self.site(other)
            self.arm_to({other: (ps + np.array([0.0, 0.0, 0.05]), Rs)}, timeout=3.0, tol=0.01, frame="base")
            return other
        return None

    def wipe(self, T, ends=(0, 1), deadline=None):
        """Wipe and sanitize table T's top from its free ends: lanes along the table pulled towards the robot, each
        hand taking the lanes on its own side (the cloth handed over between them). Returns the ends wiped (an end
        is not started after `deadline`); the cloth goes back into the bucket either way."""
        done = []
        side = self.cloth_take()
        if side is None:
            return done
        t = RM.tables()[T]
        (cx, cy), (sx, sy) = t["center"], t["size"]
        lx, ly, _ = O.CLOTH["size"]
        zt = RM.TABLE_TOP_Z
        nl = int(math.ceil((sy - 0.02) / (ly - 0.02)))
        lanes = np.linspace(cy - sy / 2 + ly / 2 - 0.01, cy + sy / 2 - ly / 2 + 0.01, nl)
        for ei, spot in enumerate(self.end_spots(T, gap=self.WIPE_GAP)):
            if ei not in ends:
                continue
            if (deadline is not None and self.t > deadline) or self.out_of_time(90.0):
                break
            side = next((sd for sd in ("left", "right") if self.held[sd] == "cloth"), None)
            if side is None:
                break
            done.append(ei)
            self.approach(*spot, h=self.WIPE_H, pitch=self.WIPE_PITCH)
            yaw = spot[2]
            hd = np.array([math.cos(yaw), math.sin(yaw)])
            lft = np.array([-math.sin(yaw), math.cos(yaw)])
            edge_x = cx - hd[0] * sx / 2
            rel = {float(y_): float((y_ - spot[1]) * lft[1]) for y_ in lanes}
            halves = {"left": [y_ for y_ in lanes if rel[float(y_)] >= -0.02],
                      "right": [y_ for y_ in lanes if rel[float(y_)] < 0.02]}
            order = [side, "right" if side == "left" else "left"]
            for k, hand in enumerate(order):
                if k == 1:
                    got = self.swap_cloth(T)
                    if got != hand:
                        break
                for y_ in sorted(halves[hand], key=lambda v: -abs(rel[float(v)])):
                    if self.held[hand] != "cloth":
                        break
                    far = (edge_x + hd[0] * 0.40, y_, zt)
                    near = (edge_x + hd[0] * (lx / 2 + 0.012), y_, zt)
                    if not self.press_stroke(hand, [far, near], yaw):
                        break
            side = next((sd for sd in ("left", "right") if self.held[sd] == "cloth"), None)
            if side is None:
                break
            ps, Rs = self.site(side)
            self.arm_to({side: (ps + np.array([0, 0, 0.10]), Rs)}, timeout=3.0, tol=0.02, frame="base")
            self.posture(0.79, 0.0)
            self.back_off()
        self.cloth_back()
        return done

    # ------------------------------------------------------------------------------------------ the reset
    def basket_rollups(self):
        """The clean roll-ups lying in the station's baskets now: {basket index: [names, highest first]}."""
        out = {}
        for k, bk in enumerate(RM.BASKETS):
            (bxc, byc), (bsx, bsy), bh = bk["center"], bk["size"], bk["h"]
            z0 = RM.STATION["box"][5]
            lst = []
            for n in self.c.layout["rollups"]:
                if n in self.c.shift.parked or n in self.held.values():
                    continue
                p, _ = self.obj(n)
                if abs(p[0] - bxc) < bsx / 2 + 0.01 and abs(p[1] - byc) < bsy / 2 + 0.01 and p[2] < z0 + bh + 0.06:
                    lst.append((p[2], -abs(p[0] - (RM.STATION["box"][0])), n))
            out[k] = [n for *_r, n in sorted(lst, reverse=True)]
        return out

    def seat_done(self, seat):
        from .scoring import rollup_ok
        for n in self.c.layout["rollups"]:
            if n not in self.c.shift.parked and n not in self.held.values() and rollup_ok(self.c, n, seat):
                return True
        return False

    def reset(self, T, deadline=None):
        """A roll-up at every seat of table T, two a trip from the station's baskets (no trip started after
        `deadline`). True when every seat has one. No roll-up is left in a hand: one that could not be laid goes
        back into its basket."""
        ok = self._reset(T, deadline)
        if any(self.held[sd] and self.held[sd] != "cart" and self.kind(self.held[sd]) == "rollup"
               for sd in ("left", "right")):
            self.free_hands()
        return ok

    def _reset(self, T, deadline=None):
        t = RM.tables()[T]
        for rnd in range(6):
            if self.out_of_time(40.0) or (deadline is not None and self.t > deadline and rnd > 0):
                return all(self.seat_done(st) for st in t["seats"])
            todo = [st for st in t["seats"] if not self.seat_done(st)]
            if not todo:
                self.free_hands()
                return True
            avail = self.basket_rollups()
            if not any(avail.values()) and not any(
                    self.held[sd] and self.held[sd] != "cart" and self.kind(self.held[sd]) == "rollup"
                    for sd in ("left", "right")):
                self.log("[ref] reset %s: no clean roll-up left" % T)
                return False
            # the end spot nearer the seats still to do, and the hands for them
            spots = self.end_spots(T)
            spot = min(spots, key=lambda sp: sum(math.hypot(st["rollup"][0] - sp[0], st["rollup"][1] - sp[1])
                                                 for st in todo[:2]))
            base = np.array(spot)
            todo.sort(key=lambda st: math.hypot(st["rollup"][0] - spot[0], st["rollup"][1] - spot[1]))
            # (a roll-up still in a hand from the last trip is laid first, without a trip to the station: bowing
            # over the baskets with it would knock it out of the hand)
            carried = {sd: self.held[sd] for sd in ("left", "right")
                       if self.held[sd] and self.held[sd] != "cart" and self.kind(self.held[sd]) == "rollup"}
            plan = {}
            for st in todo[:(len(carried) if carried else 2)]:
                sd = self.side_of(base, st["rollup"])
                free_sides = list(carried) if carried else ["left", "right"]
                if sd in plan or sd not in free_sides:
                    sd = next((o for o in free_sides if o not in plan), None)
                if sd is None:
                    break
                plan[sd] = st
            took = dict(carried)
            if not carried:
                # take them from the station's baskets (facing east: the left hand at the northern basket)
                x0 = RM.STATION["box"][0]
                yb = 0.5 * (RM.BASKETS[0]["center"][1] + RM.BASKETS[1]["center"][1])
                if not self.approach(x0 - 0.21, yb, 0.0, back=0.30, h=0.79, pitch=0.30, high=True):
                    self.log("[ref] reset %s: the station is out of reach" % T)
                    self.posture(0.79, 0.0)
                    return all(self.seat_done(st) for st in t["seats"])
                for sd in sorted(plan):
                    order = [1, 0] if sd == "left" else [0, 1]
                    done = False
                    for k in order:
                        for n in self.basket_rollups()[k]:
                            if self.pick(n, sd):
                                took[sd] = n
                                done = True
                                break
                        if done:
                            break
                # (clear of the baskets, straight up, before the body straightens: a roll-up's end dipping back into
                # its basket snags on the wall; one that does not come up clear is let go back into its basket)
                rolls = tuple(self.c.layout["rollups"])
                rim = RM.STATION["box"][5] + RM.BASKETS[0]["h"]
                for sd in list(took):
                    ps, Rs = self.site(sd)
                    up = ps + np.array([0.0, 0.0, 0.10])
                    if not self.lower(sd, up, Rs, touch_N=25.0):
                        # (the straight line at a fixed attitude runs out of the arm's reach: the rest of the way up
                        # with the hand's attitude giving a little)
                        need = rim + 0.03 - self.low_point(took[sd])
                        if need > 0.0:
                            self.lift(sd, need, allow=rolls)
                    if self.low_point(took[sd]) < rim + 0.015:
                        self.log("[ref] reset %s: %s did not come up clear of its basket: let go" % (T, took[sd]))
                        self.hand(sd, False)
                        self.step(40)
                        pu, Ru = self.site(sd)
                        self.lower(sd, pu + np.array([0.0, 0.0, 0.06]), Ru)
                        del took[sd]
                self.posture(0.79, 0.0)
                # (back from the station before the hands come to their carry pose: it lies over the baskets' rims)
                self.back_off(0.25)
                if not took:
                    self.home()
                    return False
                self.carry(z=1.05)
            plan = {sd: st for sd, st in plan.items() if sd in took}
            sx_, sy_ = spot[0], spot[1]
            tc = t["center"]
            gap_spot = [sp for sp in self.end_spots(T, gap=self.WIPE_GAP) if math.hypot(sp[0] - sx_, sp[1] - sy_) < 0.2][0]
            self.approach(*gap_spot, h=self.WIPE_H, pitch=self.WIPE_PITCH)
            hd = np.array([math.cos(gap_spot[2]), math.sin(gap_spot[2])])
            for sd, st in sorted(plan.items()):
                n = took[sd]
                x, y, ax = st["rollup"]
                # (the rule: its middle 5-14 cm in from the edge, within 7 cm of the seat's middle: 2 cm further in
                # and 5 cm nearer the robot's end of the table first; elsewhere in that window when something left
                # on the top, or the arm's reach, rules that out)
                inward = np.array([tc[0] - x, tc[1] - y])
                inward = inward - hd * (inward @ hd)
                inward = inward / max(1e-9, np.linalg.norm(inward))
                other = "right" if sd == "left" else "left"
                if self.held[other]:
                    self.hold_aside(other)
                for din, dal in ((0.02, -0.05), (0.0, -0.05), (0.04, -0.05), (0.02, -0.02), (0.0, -0.02),
                                 (0.04, -0.02), (0.02, 0.01), (0.02, -0.06)):
                    if self.held[sd] != n:
                        break
                    px_, py_ = x + inward[0] * din + hd[0] * dal, y + inward[1] * din + hd[1] * dal
                    if self.place(sd, (px_, py_, RM.TABLE_TOP_Z + 0.001), yaws=[ax, ax + math.pi], above=0.06):
                        break
            self.posture(0.79, 0.0)
            # (back from the table's end before the arms come down: hands just over the top would sweep its edge)
            self.back_off()
            self.home()
            for sd in ("left", "right"):
                if self.held[sd] and self.held[sd] != "cart":
                    self.log("[ref] reset %s: still holding %s" % (T, self.held[sd]))
        if all(self.seat_done(st) for st in t["seats"]):
            self.free_hands()
            return True
        return False

    def rollup_back(self, side):
        """A spare roll-up back into its basket at the station (the left hand's the northern one)."""
        name = self.held[side]
        if not name or name == "cart" or self.kind(name) != "rollup":
            return False
        x0 = RM.STATION["box"][0]
        yb = 0.5 * (RM.BASKETS[0]["center"][1] + RM.BASKETS[1]["center"][1])
        self.carry([side], z=1.05)
        if not self.approach(x0 - 0.21, yb, 0.0, back=0.30, h=0.79, pitch=0.15, high=True):
            return False
        bk = RM.BASKETS[1 if side == "left" else 0]
        (bxc, byc), bh = bk["center"], bk["h"]
        z0 = RM.STATION["box"][5]
        # (its length along the basket's long side, the way the roll-ups lie there)
        ok = self.place(side, (bxc, byc, z0 + bh + 0.03), yaws=[math.pi / 2, -math.pi / 2], drop=0.03, above=0.06,
                        clear_z=z0 + bh + 0.02)
        self.posture(0.79, 0.0)
        self.back_off(0.25)
        self.home()
        return ok

    def free_hands(self, keep=()):
        """Put away whatever the hands hold (but the kinds in `keep`): a guest's thing into the tub, the cloth into
        its bucket, a roll-up back into its basket. True when both hands are free of all else (the cart's handle
        aside)."""
        for sd in ("left", "right"):
            n = self.held[sd]
            if not n or n == "cart" or self.kind(n) in keep:
                continue
            if n == "cloth":
                self.cloth_back()
            elif self.kind(n) == "rollup":
                self.rollup_back(sd)
            elif n in self.c.layout["loose"]:
                if not self.c.cart.grasp:
                    self.load([sd])
        return not any(self.held[sd] and self.held[sd] != "cart" and self.kind(self.held[sd]) not in keep
                       for sd in ("left", "right"))

    # ------------------------------------------------------------------------------------------ the spill
    CONE_R = 0.50                  # m: the planner's footprint while the robot carries the cone in front of it

    def _clear(self, p, r, extra=(), cone=True):
        """The floor point p lies at least r from every footprint on the floor (and r inside the walls; the cone's
        own footprint counted unless cone=False)."""
        x, y = float(p[0]), float(p[1])
        if not (r < x < RM.ROOM_X - r and r < y < RM.ROOM_Y - r):
            return False
        obs = self.obstacles()
        if not cone:
            obs = obs[:1] if not self.c.cart.grasp else []
        for (x0, y0, x1, y1) in floor_boxes(extra=list(extra) + obs):
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - y, 0.0, y - y1)
            if math.hypot(dx, dy) < r:
                return False
        return True

    def spill_box(self):
        sp = self.c.process.spill
        xy = sp.xy
        return xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()

    def cone_spot(self):
        """Where the wet-floor cone goes: 0.6-1.1 m from the spill's middle, its base 0.32 m clear of every spill
        cell, in free floor with room for the robot to set it down; away from the party the spill is by."""
        sp = self.c.process.spill
        c = sp.center
        tc = np.array(RM.tables()[self.c.layout["spill"]["table"]]["center"], float)
        # (not in the corridor the mopping needs: the robot mops along x from the spill's east side, the head swept
        # over its whole width)
        sx0, sy0, sx1, sy1 = self.spill_box()
        corridor = (sx0 - 0.35, sy0 - 0.45, sx1 + 1.35, sy1 + 0.45)
        # (it stands there until the floor is dry, while the shift goes on: off the cart's lanes and the robot's
        # stances at the tables' ends where it can, in open floor, 0.30 m round it and room for the robot beside it;
        # squeezed in only if nothing else is free)
        bands = self._cart_bands()
        stances = [sp_[:2] for T_ in RM.tables() for g_ in (WORK_GAP, self.WIPE_GAP) for sp_ in self.end_spots(T_, g_)]

        def in_the_way(p):
            for (x0, y0, x1, y1) in bands:
                if x0 - 0.20 < p[0] < x1 + 0.20 and y0 - 0.20 < p[1] < y1 + 0.20:
                    return True
            return any(math.hypot(p[0] - q[0], p[1] - q[1]) < 0.60 for q in stances)

        for r_cone, r_st, way in ((0.30, ROBOT_R + 0.04, True), (0.30, ROBOT_R + 0.04, False),
                                  (0.22, ROBOT_R - 0.04, False)):
            best = None
            # (1.15 m only squeezed in: the rule allows 1.2 m from the spill's middle)
            for r in ((0.75, 0.9, 1.05, 0.62) if r_cone > 0.25 else (0.75, 0.9, 1.05, 0.62, 1.15)):
                for a in np.linspace(-math.pi, math.pi, 24, endpoint=False):
                    u = np.array([math.cos(a), math.sin(a)])
                    p = c + r * u
                    if float(np.min(np.linalg.norm(sp.xy - p[None, :], axis=1))) < 0.32:
                        continue
                    if not self._clear(p, r_cone):
                        continue
                    if way and in_the_way(p):
                        continue
                    if corridor[0] - 0.17 < p[0] < corridor[2] + 0.17 and corridor[1] - 0.17 < p[1] < corridor[3] + 0.17:
                        continue
                    # the robot sets it down from 0.42 m behind it, facing it
                    st = p - 0.42 * u
                    if not self._clear(st, r_st) and not self._clear(p + 0.42 * u, r_st):
                        continue
                    sc = -float(np.linalg.norm(p - tc)) + 0.3 * abs(r - 0.8)
                    if best is None or sc < best[0]:
                        best = (sc, p)
            if best is not None:
                return best[1]
        return None

    def _cart_bands(self):
        """Where the cart runs (x0, y0, x1, y1): along each row's parking line, from its west-most spot to the east
        zone, and the east zone's cross lane between the lines."""
        half_l, half_w = RM.CART["size"][0] / 2 + 0.12, RM.CART["size"][1] / 2 + 0.03   # (the handle sticks out)
        lines = {}
        for T in RM.tables():
            px, py = self.cart_park(T)
            k = round(py, 3)
            lines[k] = min(px, lines.get(k, px))
        out = [(x - half_l, y - half_w, 6.75 + half_l, y + half_w) for y, x in lines.items()]
        out.append((6.75 - half_l, min(lines) - half_w, 6.75 + half_l, max(lines) + half_w))
        return out

    def cone_out(self):
        """The wet-floor cone from its spot to beside the spill (carried by its bar, set down upright)."""
        spot = self.cone_spot()
        if spot is None:
            self.log("[ref] cone: no spot beside the spill")
            return False
        po, _ = self.obj("cone")
        # take it: standing west of it, facing the wall (its bar runs across in front of the robot)
        stance = (po[0] - 0.42, po[1], 0.0)
        self.approach(*stance, back=0.30, h=0.79, pitch=0.15, high=True)
        side = "right"
        if not self.pick("cone", side) and not self.pick("cone", "left"):
            self.log("[ref] cone: could not take it")
            return False
        side = next(sd for sd in ("left", "right") if self.held[sd] == "cone")
        ps, Rs = self.site(side)
        self.arm_to({side: (ps + np.array([0.0, 0.0, 0.10]), Rs)}, timeout=3.0, tol=0.02, frame="base")
        self.posture(0.79, 0.0)
        self.back_off(0.25)
        # to a stance facing the spot, 0.42 m before it
        b = self.base()
        best = None
        for r_st in (ROBOT_R + 0.04, ROBOT_R - 0.02):
            for a in np.linspace(-math.pi, math.pi, 16, endpoint=False):
                u = np.array([math.cos(a), math.sin(a)])
                st = spot - 0.42 * u
                # (room for the robot at the stance and at the point it walks in from, 0.25 m behind it, and for the
                # cone it carries in front of it)
                if self._clear(st, r_st) and self._clear(st - 0.25 * u, r_st) and self._clear(spot - 0.21 * u, 0.24):
                    d_ = math.hypot(st[0] - b[0], st[1] - b[1])
                    if best is None or d_ < best[0]:
                        best = (d_, st, a)
            if best is not None:
                break
        if best is None:
            self.log("[ref] cone: no stance by its spot")
            return False
        _, st, a = best
        big = Planner(self.CONE_R)
        path = big.path((b[0], b[1]), (st[0] - 0.25 * math.cos(a), st[1] - 0.25 * math.sin(a)),
                        extra=self.obstacles())
        if path is None:
            self.walk(st[0] - 0.25 * math.cos(a), st[1] - 0.25 * math.sin(a), a, carrying=True)
        else:
            self._follow(path, a)
        self.goto(st[0], st[1], a, tol=0.02, yaw_tol=0.02)
        # down to the floor in front, upright, then let go
        rel_p, rel_R = self.rel["cone"]
        po, Ro = self.obj("cone")
        target = np.array([spot[0], spot[1], 0.004])
        for dyaw in (0.0, 0.3, -0.3, 0.6, -0.6):
            Ro_ = Rz(yaw_of(Ro) + dyaw)
            Rs_ = Ro_ @ rel_R.T
            pg = target - Rs_ @ rel_p
            q, ep, er = self.ik(side, pg, Rs_)
            if ep < 0.008 and er < 0.06:
                hi = pg + np.array([0.0, 0.0, 0.06])
                self.arm_to({side: (hi, Rs_)}, timeout=4.0, tol=0.01)
                self.lower(side, pg, Rs_, touch_N=6.0)
                break
        self.hand(side, False)
        ps, Rs = self.site(side)
        self.arm_to({side: (ps + np.array([0.0, 0.0, 0.08]), Rs)}, timeout=2.0, tol=0.02, frame="base")
        self.back_off(0.25)
        self.home()
        up, held, xy = self.c.shift.cone_state()
        return bool(up and not held)

    def walk_wide(self, x, y, yaw, radius=0.52):
        """Walk planned with a wide footprint (a tool carried in front: the cone, the mop's hanging head)."""
        b = self.base()
        big = Planner(radius)
        path = big.path((b[0], b[1]), (x, y), extra=self.obstacles())
        if path is None:
            self.back_off(0.30)
            b = self.base()
            path = big.path((b[0], b[1]), (x, y), extra=self.obstacles())
        if path is None:
            self.log("[ref] walk_wide: no path to (%.2f, %.2f); a normal walk" % (x, y))
            return self.walk(x, y, yaw, carrying=True)
        self._follow(path, yaw)
        return self.goto(x, y, yaw, tol=0.02, yaw_tol=0.03)

    def _follow(self, path, yaw_end):
        prev = tuple(self.base()[:2])
        for k, (px, py) in enumerate(path):
            hd = math.atan2(py - prev[1], px - prev[0]) if math.hypot(px - prev[0], py - prev[1]) > 0.05 else \
                self.base()[2]
            if abs(wrap(hd - self.base()[2])) > 0.35:
                self.goto(prev[0], prev[1], hd, tol=0.05, yaw_tol=0.08)
            self.goto(px, py, hd, tol=0.05, yaw_tol=0.15)
            prev = (px, py)
        self.goto(prev[0], prev[1], yaw_end, tol=0.03, yaw_tol=0.05)

    MOP_GRIP_H = 1.00              # m: where the hand takes the mop's handle (above the floor, the mop upright)

    def mop_take(self):
        """The mop out of its clip: the handle taken at about the hand's height from beside the mop bucket."""
        po, Ro = self.obj("mop")
        # stand west of the handle, facing it (the wall)
        cands = []
        for a in np.linspace(-math.pi / 4, math.pi / 4, 7):
            u = np.array([math.cos(a), math.sin(a)])
            for dist in (0.42, 0.48, 0.38):
                st = po[:2] - dist * u
                if self._clear(st, ROBOT_R - 0.06):
                    cands.append((abs(a), st, a))
        if not cands:
            self.log("[ref] mop: no stance")
            return None
        cands.sort(key=lambda c_: c_[0])
        for _, st, a in cands[:4]:
            self.approach(st[0], st[1], a, back=0.30, h=0.79, pitch=0.10, high=True)
            for side in ("right", "left"):
                if self.pick("mop", side):
                    # (carried high: the head ~25 cm off the floor, so its swing as the robot turns stays clear)
                    ps, Rs = self.site(side)
                    self.lower(side, ps + np.array([0.0, 0.0, 0.14]), Rs, touch_N=30.0)
                    self.posture(0.79, 0.0)
                    self.back_off(0.30)
                    return side
        return None

    def mop_pose(self, side, sw, f, lean=0.52):
        """The grasp-site pose putting the mop's swivel at sw (room), its handle leaning back from it towards the
        robot by `lean` (rad from the vertical) in the direction -f, the head's long side across f."""
        rel_p, rel_R = self.rel["mop"]
        f = np.array([f[0], f[1], 0.0]) / max(1e-9, math.hypot(f[0], f[1]))
        up = np.array([0.0, 0.0, 1.0])
        zh = math.cos(lean) * up - math.sin(lean) * f
        xh = np.cross(up, f)
        xh /= np.linalg.norm(xh)
        yh = np.cross(zh, xh)
        Rh = np.stack([xh, yh, zh], axis=1)
        # (the head's long side runs along the handle frame's x: across the stroke)
        Rs = Rh @ rel_R.T
        ps = np.asarray(sw, float) - Rs @ rel_p
        return ps, Rs

    def mop_spill(self):
        """Mop the spill up: passes along the aisle, pushed and pulled, the head flat on the floor; then the mop back
        in its clip. Returns the share of the spill taken up."""
        side = next((sd for sd in ("left", "right") if self.held[sd] == "mop"), None)
        if side is None:
            self.cart_off_spill()
            side = self.mop_take()
        if side is None:
            return 0.0
        sp = self.c.process.spill
        for rnd in range(3):
            wet = sp.wet_cells()
            if not wet.any() or self.out_of_time(30.0) or self.held[side] != "mop":
                break
            # (the first round over the whole spill, then over what is still wet)
            box = self.spill_box() if rnd == 0 else (float(sp.xy[wet, 0].min()), float(sp.xy[wet, 1].min()),
                                                      float(sp.xy[wet, 0].max()), float(sp.xy[wet, 1].max()))
            self._mop_round(side, box)
        return sp.removed_fraction()

    MOP_LEAN = 0.22                  # rad: the handle's lean from the vertical during a pass
    MOP_REACH = 0.30                 # m: the hand ahead of the pelvis during a pass

    MOP_GRIP_L = 0.93                # m: the grasp site from the swivel along the handle (mop_take's grasp)

    def _mop_ahead(self):
        """How far the swivel runs ahead of the pelvis during a pass (the grasp as taken, else mop_take's)."""
        L = float(np.linalg.norm(self.rel["mop"][0])) if "mop" in self.rel else self.MOP_GRIP_L
        return self.MOP_REACH + L * math.sin(self.MOP_LEAN)

    @staticmethod
    def _mop_lanes(box):
        x0, y0, x1, y1 = box
        cy = 0.5 * (y0 + y1)
        ny = max(1, int(math.ceil((y1 - y0 + 0.10) / 0.20)))
        return [cy + (k - (ny - 1) / 2) * 0.20 for k in range(ny)]

    MOP_LAT = 0.14                   # m: the swivel to the hand's side of the pelvis during a pass

    @staticmethod
    def _lane_sweep(hy, box, boxes, ahead, off=0.14):
        """The head's sweep (x range) for the pass along the head's lane hy over `box` (the pelvis runs `off` to the
        other side of it): from just east of the spill to just past its west edge, trimmed where the head's sweep, the
        handle or the robot's body (its arms and a margin) would reach a chair, a table or the cart (seated guests sit
        there: neither the mop nor the robot may touch them)."""
        x0, y0, x1, y1 = box
        cx = 0.5 * (x0 + x1)
        py = hy - off
        body = ROBOT_R + 0.08
        hx_lo, hx_hi = x0 - 0.05, x1 + 0.10
        for (bx0, by0, bx1, by1) in boxes:
            in_head = not (by1 < hy - 0.21 - 0.08 or by0 > hy + 0.21 + 0.08)
            in_body = not (by1 < py - body or by0 > py + body)
            if in_head and not (bx1 < hx_lo - 0.10 or bx0 > hx_hi + ahead + ROBOT_R):
                if 0.5 * (bx0 + bx1) <= cx:
                    hx_lo = max(hx_lo, bx1 + 0.08 + 0.07)
                else:
                    hx_hi = min(hx_hi, bx0 - 0.08 - 0.07, bx0 - body - ahead)
            # (the pelvis runs from hx_hi + ahead back to hx_lo + ahead)
            if in_body and not (bx1 < hx_lo + ahead - body or bx0 > hx_hi + ahead + body):
                if 0.5 * (bx0 + bx1) <= cx + ahead:
                    hx_lo = max(hx_lo, bx1 + body - ahead)
                else:
                    hx_hi = min(hx_hi, bx0 - body - ahead)
        return hx_lo, hx_hi

    def cart_off_spill(self):
        """The parked cart out of the mopping's way, as a busser moves it before mopping: when it shortens a pass over
        the spill, it is pushed clear, along its aisle away from the spill when it stands west of it (and the aisle
        there is free), else to its home."""
        if self.c.cart.grasp:
            return True
        box = self.spill_box()
        ahead = self._mop_ahead()
        lanes = self._mop_lanes(box)
        others = list(floor_boxes(extra=self.obstacles(cart=False)))
        free = [self._lane_sweep(ly, box, others, ahead) for ly in lanes]

        def shortens(cb):
            for ly, (lo0, hi0) in zip(lanes, free):
                lo1, hi1 = self._lane_sweep(ly, box, others + [cb], ahead)
                if lo1 > lo0 + 0.03 or hi1 < hi0 - 0.03:
                    return True
            return False

        def hits(b, segs, against):
            """The cart's footprint b, moved along the segments (dx, dy steps from its place), meets one of `against`."""
            ox, oy = 0.0, 0.0
            for (dx, dy) in segs:
                n = max(1, int(math.hypot(dx, dy) / 0.08))
                for k in range(1, n + 1):
                    u, v = ox + dx * k / n, oy + dy * k / n
                    for (a0, b0, a1, b1) in against:
                        if b[0] + u < a1 and b[2] + u > a0 and b[1] + v < b1 and b[3] + v > b0:
                            return True
                ox, oy = ox + dx, oy + dy
            return False

        cb = self.cart_box()
        if not shortens(cb):
            return True
        q = self.cart_pose()
        x0, y0, x1, y1 = box
        self.log("[ref] the cart (%.2f, %.2f) is in the mopping's way" % (q[0], q[1]))
        if 0.5 * (cb[0] + cb[2]) <= 0.5 * (x0 + x1):
            sh = (x0 - 0.30) - cb[2]
            nb = (cb[0] + sh, cb[1], cb[2] + sh, cb[3])
            if nb[0] > 0.30 and not shortens(nb) and not hits(cb, [(sh, 0.0)], others):
                if self.cart_grip() and self.cart_go(q[0] + sh, q[1]):
                    self.cart_release()
                    return True
                self.cart_release()
        hx, hy, _ = RM.CART["home"]
        q = self.cart_pose()
        cb = self.cart_box()
        segs = []
        if q[0] < 6.75 - 0.05:
            segs.append((6.75 - q[0], 0.0))
        segs.append((hx - max(q[0], 6.75), hy - q[1]))
        cone = self.obstacles(cart=False)
        if not shortens((cb[0] + hx - q[0], cb[1] + hy - q[1], cb[2] + hx - q[0], cb[3] + hy - q[1])) and \
                not hits(cb, segs, cone):
            return self.cart_home()
        self.log("[ref] the cart stays where it is")
        return False

    def _mop_round(self, side, box):
        """Out-and-back passes along x (the aisles run east-west) over `box`, lanes 0.20 m apart across it."""
        sp = self.c.process.spill
        x0, y0, x1, y1 = box
        lanes = self._mop_lanes(box)
        z_sw = 0.046 - 0.004                     # the swivel's height with the pad flat, pressed 4 mm into its give
        lean = self.MOP_LEAN
        ahead = self._mop_ahead()                # the swivel ahead of the pelvis
        # (facing west the right hand carries the swivel MOP_LAT to the north of the pelvis, the left to the south)
        off = self.MOP_LAT if side == "right" else -self.MOP_LAT
        for dirn in (-1.0,):
            f = np.array([dirn, 0.0])
            boxes = floor_boxes(extra=self.obstacles())
            for hy in lanes:
                if self.out_of_time(30.0):
                    break
                ly = hy - off                    # the pelvis's lane
                hx_lo, hx_hi = self._lane_sweep(hy, box, boxes, ahead, off)
                if hx_lo > x0 - 0.05 + 1e-6 or hx_hi < x1 + 0.10 - 1e-6:
                    self.log("[ref] mop: lane %.2f trimmed to sweep %.2f..%.2f (spill x %.2f..%.2f)" % (
                        ly, hx_lo, hx_hi, x0, x1))
                if hx_hi - hx_lo < 0.10:
                    self.log("[ref] mop: lane %.2f blocked (sweep %.2f..%.2f, ahead %.2f, spill x %.2f..%.2f)" % (
                        ly, hx_lo, hx_hi, ahead, x0, x1))
                    continue
                start = np.array([hx_hi + ahead, ly])
                if not self._clear(start, ROBOT_R - 0.02):
                    continue
                if not self.walk_wide(start[0] + 0.30, start[1], math.pi):
                    self.log("[ref] mop: no way to lane %.2f" % hy)
                    continue
                self.goto(start[0], start[1], math.pi, tol=0.02, yaw_tol=0.02)
                b = self.base()
                hd = np.array([math.cos(b[2]), math.sin(b[2])])
                lat = np.array([-hd[1], hd[0]]) * (0.14 if side == "left" else -0.14)   # (on the hand's side)
                # lean it in the air (the head clear of the floor), then straight down until the pad presses
                sw = np.array([b[0] + hd[0] * ahead + lat[0], b[1] + hd[1] * ahead + lat[1], z_sw + 0.20])
                ps, Rs = self.mop_pose(side, sw, hd, lean)
                self.move_smooth(side, ps, Rs)
                self.step(60)                    # (the head settles flat on its swivel)
                if not self.mop_down(side, Rs, press_N=3.5):
                    self.log("[ref] mop: the head did not get down at lane %.2f" % ly)
                    ps_, Rs_ = self.site(side)
                    self.move_smooth(side, ps_ + np.array([0.0, 0.0, 0.12]), Rs_)
                    continue
                ps, Rs = self.site(side)
                # carry the hand with the base: the pass out and back
                pb, Rb = self.to_base(ps, Rs)
                self.c.command({f"{side}_arm": {"pose": [float(v) for v in pb] + [float(v) for v in quat_of(Rb)],
                                                "frame": "base", "path": "line"}})
                end_x = hx_lo + ahead                # the pelvis when the head's middle is past the west edge
                pr = self.c.process
                for xa, xb in ((start[0], end_x), (end_x, start[0])):
                    n_ = max(1, int(abs(xb - xa) / 0.05))
                    for k_ in range(1, n_ + 1):
                        self.goto(xa + (xb - xa) * k_ / n_, ly, math.pi, tol=0.04, yaw_tol=0.05, timeout=6.0)
                        if self.held[side] != "mop" or self.c.grips.holding(side) != "mop":
                            break
                        _, N = pr._contact_force(pr.pad_geom, {pr.floor_geom: "floor"})
                        if N > 9.0 or N < 2.5:
                            # (the press drifted: the carried hand 3 mm up or down)
                            pb = pb + np.array([0.0, 0.0, 0.003 if N > 9.0 else -0.003])
                            self.c.command({f"{side}_arm": {"pose": [float(v) for v in pb] +
                                                            [float(v) for v in quat_of(Rb)], "frame": "base",
                                                            "path": "line"}})
                # head up for the next lane (upright if the lift does not come out: never walked with the head down)
                ps_, Rs_ = self.site(side)
                if not self.move_smooth(side, ps_ + np.array([0.0, 0.0, 0.12]), Rs_) and \
                        self.held[side] == "mop" and not self.mop_upright(side):
                    self.log("[ref] mop: the head does not come up: the mopping ends here")
                    break
                if self.held[side] != "mop" or not self.sync_held():
                    break

    def move_smooth(self, side, p, R, dpos=0.01, drot=0.05, wait=30):
        """To the grasp-site pose (p, R) in small steps (at most dpos m and drot rad each), IK on the measured state
        from the last step's joints, each step waited for: a long tool in the hand moves slowly and in one piece."""
        cd = self.chk.d
        cd.qpos[:] = self.d.qpos
        p0, R0 = self.site(side)
        p = np.asarray(p, float)
        e = R0.T @ np.asarray(R, float)
        ang = math.acos(float(np.clip((np.trace(e) - 1) / 2, -1, 1)))
        n = max(1, int(math.ceil(max(float(np.linalg.norm(p - p0)) / dpos, ang / drot))))
        q0 = np.zeros(4)
        mujoco.mju_mat2Quat(q0, R0.reshape(-1))
        q1 = quat_of(R)
        if float(q0 @ q1) < 0:
            q1 = -q1
        qk = self.d.qpos[self.r.arm_q[side]].copy()
        for k in range(1, n + 1):
            u = k / n
            # (normalised linear interpolation of the attitude: steps of a few degrees)
            qu = q0 * (1 - u) + q1 * u
            qu /= np.linalg.norm(qu)
            pk = p0 + (p - p0) * u
            q, ep, er = self.nik.solve(cd.qpos, side, pk, qu, qk, iters=30, w_rot=0.35, natural=True, null_total=0.01)
            if ep > 0.008 or float(np.max(np.abs(q - qk))) > 0.25:
                self.log("[ref] move_smooth %s: stopped at step %d of %d (ik %.3f)" % (side, k, n, ep))
                return False
            qk = q
            self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
            for _ in range(wait):
                self.step(1)
                if float(np.max(np.abs(self.d.qpos[self.r.arm_q[side]] - q))) < 0.01:
                    break
        return True

    def mop_upright(self, side, z_sw=0.25):
        """The mop held upright for carrying: its handle vertical in front of the hand's shoulder, the head ~20 cm
        off the floor (its long side across the robot's heading)."""
        rel_p, rel_R = self.rel["mop"]
        b = self.base()
        hd = np.array([math.cos(b[2]), math.sin(b[2])])
        lat = np.array([-hd[1], hd[0]]) * (0.14 if side == "left" else -0.14)
        sw = np.array([b[0] + hd[0] * 0.30 + lat[0], b[1] + hd[1] * 0.30 + lat[1], z_sw])
        Rh = Rz(b[2] + math.pi / 2)
        Rs = Rh @ rel_R.T
        ps = sw - Rs @ rel_p
        return self.move_smooth(side, ps, Rs)

    def mop_down(self, side, R, press_N=5.0, step=0.004, max_drop=0.36):
        """Lower the mop straight down (the hand's attitude kept, joint targets from IK on the measured state) until
        its pad presses the floor with press_N."""
        pr = self.c.process
        cd = self.chk.d
        # (the arm settled where it was sent first: the descent starts from where the hand really is)
        qc = np.array(self.c.ctrl.arm_cmd[side], float)
        for _ in range(60):
            self.step(1)
            if float(np.max(np.abs(self.d.qpos[self.r.arm_q[side]] - qc))) < 0.006:
                break
        cd.qpos[:] = self.d.qpos
        p0, _ = self.site(side)
        qk = self.d.qpos[self.r.arm_q[side]].copy()
        qt = quat_of(R)
        n = int(max_drop / step)
        for k in range(1, n + 1):
            pk = p0 - np.array([0.0, 0.0, step * k])
            q, ep, er = self.nik.solve(cd.qpos, side, pk, qt, qk, iters=30, w_rot=0.35, natural=True, null_total=0.01)
            if ep > 0.006 or float(np.max(np.abs(q - qk))) > (0.4 if k == 1 else 0.2):
                self.log("[ref] mop_down: stopped at %.3f (ik ep %.4f er %.3f jump %.3f) q %s" % (
                    pk[2], ep, er, float(np.max(np.abs(q - qk))), np.round(q, 2)))
                return False
            qk = q
            self.c.command({f"{side}_arm": {"q": [float(v) for v in q]}})
            for _ in range(12):
                self.step(1)
                _, N = pr._contact_force(pr.pad_geom, {pr.floor_geom: "floor"})
                if N >= press_N:
                    # (on the floor: the arm held where it is, then eased up while it presses too hard)
                    qh = self.d.qpos[self.r.arm_q[side]].copy()
                    self.c.command({f"{side}_arm": {"q": [float(v) for v in qh]}})
                    self.step(25)
                    for _k in range(6):
                        _, N = pr._contact_force(pr.pad_geom, {pr.floor_geom: "floor"})
                        if N < 9.0:
                            break
                        cd.qpos[:] = self.d.qpos
                        pu = self.site(side)[0] + np.array([0.0, 0.0, 0.003])
                        qh, ep, er = self.nik.solve(cd.qpos, side, pu, qt, qh, iters=30, w_rot=0.35, natural=True,
                                                    null_total=0.01)
                        self.c.command({f"{side}_arm": {"q": [float(v) for v in qh]}})
                        self.step(15)
                    return True
                if float(np.max(np.abs(self.d.qpos[self.r.arm_q[side]] - q))) < 0.005:
                    break
        return False

    def mop_home(self):
        """The mop back in its wall clip: the handle upright through the clip's jaw, let go."""
        side = next((sd for sd in ("left", "right") if self.held[sd] == "mop"), None)
        if side is None:
            return False
        jaw = np.array(RM.MOP_HOME["clip"], float)
        rel_p, rel_R = self.rel["mop"]
        self.mop_upright(side)
        cands = []
        for a in np.linspace(-math.pi / 4, math.pi / 4, 7):
            u = np.array([math.cos(a), math.sin(a)])
            for dist in (0.42, 0.48, 0.38):
                st = jaw[:2] - dist * u
                if self._clear(st, ROBOT_R - 0.06):
                    cands.append((abs(a), st, a))
        cands.sort(key=lambda c_: c_[0])
        for _, st, a in cands[:4]:
            self.walk_wide(st[0] - 0.30 * math.cos(a), st[1] - 0.30 * math.sin(a), a)
            self.goto(st[0], st[1], a, tol=0.012, yaw_tol=0.015)
            # the handle vertical, its axis through the jaw: the swivel straight below the jaw, the head along the
            # wall and just off the floor; a step out from the wall first, then in
            for yaw_h in (math.pi / 2, -math.pi / 2):
                Rh = Rz(yaw_h)
                Rs = Rh @ rel_R.T
                sw = np.array([jaw[0], jaw[1], 0.075])
                ps = sw - Rs @ rel_p
                q, ep, er = self.ik(side, ps, Rs)
                if ep > 0.006 or er > 0.05:
                    continue
                out_ = ps - np.array([0.10, 0.0, 0.0])
                if not self.move_smooth(side, out_ + np.array([0.0, 0.0, 0.03]), Rs):
                    continue
                self.step(80)                    # (the head stops swinging)
                self.move_smooth(side, ps + np.array([0.0, 0.0, 0.03]), Rs, dpos=0.005)
                self.move_smooth(side, ps, Rs, dpos=0.005)
                self.hand(side, False)
                self.step(30)
                ps_, Rs_ = self.site(side)
                self.arm_to({side: (ps_ - 0.06 * np.array([math.cos(a), math.sin(a), 0.0]), Rs_)}, timeout=2.0,
                            tol=0.02)
                self.back_off(0.25)
                self.home()
                return bool(self.c.clips.engaged.get("mop"))
        self.log("[ref] mop_home: no pose at the clip")
        return False

    def cone_home(self):
        """The cone back on its spot by the wall (once the floor is dry)."""
        po, _ = self.obj("cone")
        cands = []
        for a in np.linspace(-math.pi, math.pi, 12, endpoint=False):
            u = np.array([math.cos(a), math.sin(a)])
            st = po[:2] - 0.42 * u
            if self._clear(st, ROBOT_R - 0.04, cone=False):
                cands.append((abs(wrap(a - math.pi / 2)), st, a))
        cands.sort(key=lambda c_: c_[0])
        if not cands:
            return False
        _, st, a = cands[0]
        self.approach(st[0], st[1], a, back=0.30, h=0.79, pitch=0.15, high=True)
        side = None
        for sd in ("right", "left"):
            if self.pick("cone", sd):
                side = sd
                break
        if side is None:
            return False
        ps, Rs = self.site(side)
        self.arm_to({side: (ps + np.array([0.0, 0.0, 0.10]), Rs)}, timeout=3.0, tol=0.02, frame="base")
        self.posture(0.79, 0.0)
        self.back_off(0.25)
        hx, hy = RM.CONE_HOME
        st = np.array([hx - 0.42, hy])
        b = self.base()
        big = Planner(self.CONE_R)
        path = big.path((b[0], b[1]), (st[0] - 0.25, st[1]), extra=self.obstacles())
        if path is None:
            self.walk(st[0] - 0.25, st[1], 0.0, carrying=True)
        else:
            self._follow(path, 0.0)
        self.goto(st[0], st[1], 0.0, tol=0.02, yaw_tol=0.02)
        rel_p, rel_R = self.rel["cone"]
        po, Ro = self.obj("cone")
        Rs_ = Ro @ rel_R.T
        Rs_ = Rz(yaw_of(Ro)) @ rel_R.T
        pg = np.array([hx, hy, 0.004]) - Rs_ @ rel_p
        self.arm_to({side: (pg + np.array([0.0, 0.0, 0.06]), Rs_)}, timeout=4.0, tol=0.01)
        self.lower(side, pg, Rs_, touch_N=6.0)
        self.hand(side, False)
        ps, Rs = self.site(side)
        self.arm_to({side: (ps + np.array([0.0, 0.0, 0.08]), Rs)}, timeout=2.0, tol=0.02, frame="base")
        self.back_off(0.25)
        self.home()
        return self.c.shift.cone_home()

    # ------------------------------------------------------------------------------------------ the shift
    EST = {"bus_round": 75.0, "wipe_end": 160.0, "reset_trip": 110.0, "dock": 110.0, "home": 70.0,
           "cone_out": 70.0}

    def tub_count(self):
        return len(self.tub_contents())

    def run_shift(self):
        """The whole shift: the tables as their parties leave (cleared, wiped, reset), the dish crew at the dock
        when the tub fills, the spill (the cone at once, mopped, the mop back; the cone home once the floor is dry),
        then closing (the last dock trip, the cart home, the cloth in its bucket)."""
        lay = self.c.layout
        sp_t = float(lay["spill"]["t"])
        sp = self.c.process.spill
        cap = self.c.sim_cap
        jobs = {}
        for T, st in lay["tables"].items():
            if st["state"] in ("dirty", "occupied"):
                jobs[T] = {"leave": 0.0 if st["state"] == "dirty" else float(st["leave_s"]), "stage": "bus",
                           "ends": [0, 1], "fails": 0}
        spill = "pending"
        dock_retry_at = 0.0
        closing_at = cap - (self.EST["dock"] + self.EST["home"] + 40.0)
        self.log("[shift] tables %s spill at %.0f" % ({T: round(j["leave"]) for T, j in jobs.items()}, sp_t))

        def free_until():
            """The time by which the next piece of work must end (the spill's start while it is pending)."""
            lim = closing_at
            if spill == "pending":
                lim = min(lim, sp_t + 25.0)
            return lim

        while not self.out_of_time(30.0):
            t = self.t
            # ---- the spill: the cone out at once, then the mop
            if spill == "pending" and t >= sp_t - 1.0:
                self.log("[shift] spill at t %.0f" % t)
                while not sp.active and not self.out_of_time(30.0):
                    self.step(20)
                self.cone_out()
                self.mop_spill()
                self.mop_home()
                spill = "mopped"
                continue
            if spill == "mopped" and sp.dry_at is not None and t > sp.dry_at + 5.0 and t < closing_at - 90.0:
                self.cone_home()
                spill = "done"
                continue
            if t >= closing_at:
                break
            # ---- the next piece of table work
            ready = [T for T, j in jobs.items() if j["stage"] != "done" and t >= j["leave"] + 2.0]
            ready.sort(key=lambda T: (jobs[T]["leave"], T))
            lim = free_until()
            did = False
            for T in ready:
                j = jobs[T]
                # (the hands free of anything the next piece of work does not use: carried along, it is knocked
                # out of a hand at the first bow)
                self.free_hands(keep={"bus": (), "wipe": ("cloth",), "reset": ("rollup",)}.get(j["stage"], ()))
                if j["stage"] == "bus":
                    if self.tub_count() >= 14 and t >= dock_retry_at:
                        if t + self.EST["dock"] <= lim:
                            if not self.to_dock():
                                # (the cart did not get there: on with the work, another try in a while)
                                dock_retry_at = self.t + 150.0
                                self.log("[shift] the dock trip failed at t %.0f" % self.t)
                            did = True
                        break
                    if t + self.EST["bus_round"] > lim:
                        break
                    n0 = len(self.table_items(T))
                    done = self.bus(T, deadline=lim - self.EST["bus_round"])
                    n1 = len(self.table_items(T))
                    if done or not n1 or (n1 == n0 and self.t < lim - 5.0):
                        # (cleared, or nothing more it can take: on to the wipe)
                        j["stage"] = "wipe"
                    did = True
                    break
                if j["stage"] == "wipe":
                    if t + self.EST["wipe_end"] > lim:
                        break
                    got = self.wipe(T, ends=j["ends"], deadline=lim - self.EST["wipe_end"])
                    j["ends"] = [e for e in j["ends"] if e not in got]
                    if not j["ends"] or not got:
                        j["stage"] = "reset"
                    did = True
                    break
                if j["stage"] == "reset":
                    if t + self.EST["reset_trip"] > lim:
                        break
                    n0 = sum(1 for st_ in RM.tables()[T]["seats"] if self.seat_done(st_))
                    if self.reset(T, deadline=lim - self.EST["reset_trip"]):
                        j["stage"] = "done"
                    elif self.out_of_time(60.0):
                        j["stage"] = "done"
                    else:
                        n1 = sum(1 for st_ in RM.tables()[T]["seats"] if self.seat_done(st_))
                        j["fails"] = 0 if n1 > n0 else j["fails"] + 1
                        if j["fails"] >= 2:
                            self.log("[shift] %s: the reset makes no progress; left as it is" % T)
                            j["stage"] = "done"
                    did = True
                    break
            if not did:
                # (nothing that fits before the spill, or no party has left yet: wait by the cone if the spill is
                # next, else at the station)
                if spill == "pending" and t >= sp_t - 300.0:
                    po, _ = self.obj("cone")
                    st_ = (po[0] - 0.72, po[1], 0.0)
                    b = self.base()
                    if math.hypot(b[0] - st_[0], b[1] - st_[1]) > 0.2:
                        self.walk(*st_)
                    self.step(int(min(20.0, max(1.0, sp_t - self.t)) / TICK))
                else:
                    nxt = [j["leave"] for j in jobs.values() if j["stage"] != "done" and j["leave"] > t]
                    if not nxt and spill != "mopped":
                        break
                    self.step(int(10.0 / TICK))
        # ---- closing
        self.log("[shift] closing at t %.0f (tub %d)" % (self.t, self.tub_count()))
        self.free_hands()
        if spill == "mopped" and sp.dry_at is not None and self.t < cap - 200.0:
            self.cone_home()
        if self.tub_count() > 0 and self.t < cap - self.EST["dock"] - self.EST["home"]:
            self.to_dock()
        if self.t < cap - self.EST["home"]:
            self.cart_home()
        return {T: j["stage"] for T, j in jobs.items()}

    # ------------------------------------------------------------------------------------------ the dock
    def dock_point(self):
        x0, y0, x1, y1 = RM.DOCK["box"]
        return ((x0 + x1) / 2 - 0.08, (y0 + y1) / 2)

    def to_dock(self, wait=True):
        dx, dy = self.dock_point()
        if not self.cart_route(dx, dy):
            return False
        self.cart_release()
        b = self.base()
        self.goto(b[0] + 0.25, b[1], b[2], tol=0.03)
        if wait:
            t0 = self.t
            while not self.c.shift.dock_served and self.t - t0 < RM.DOCK["crew_after_s"] + 8.0:
                self.step(50)
        return self.c.shift.dock_served

    def cart_home(self):
        hx, hy, _ = RM.CART["home"]
        ok = self.cart_route(hx, hy)
        self.cart_release()
        return ok
