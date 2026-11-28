"""The lunch service around the robot (cell internals; the cell manual states the rules, never a shift's numbers): the
bus cart's casters and its servo hold, the guests leaving, the floor spill, the dish crew at the dock, and the bodies
the cell keeps out of play (parked far north of the room).

The cart: while a hand holds its push handle the cell drives it (its two slides and its turn) towards the pose the
holding hands imply (each hand's grasp point on the handle as it was when the hand closed; with two hands, their
midpoint and the line between them), with a servo whose force per holding hand is the hands' load (physics.Grips:
30 N per hand, 50 ms over: the handle slips); a hand that moves its grasp off the handle (CART_HANDLE_OFF) lets go.
Not held, the casters' brakes hold the cart (a stiff damper on its slides and turn).

The dish crew: a cart left standing in the dock (its whole footprint inside the taped box, its long side along the
pass within 12 deg, no hand on it) for RM.DOCK["crew_after_s"] is served once: the crew takes everything that lies in
the tub (the guests' tableware and linen, roll-ups) and the crumbs in it and changes the sanitizer (the cloth, if it is in the bucket, comes back wet with fresh sanitizer). The crew serves a cart again
only after it has left the dock.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import objects as O
from . import physics as PH
from . import room as RM
from .layout import PARK_AT

TICK = 0.01


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class CartServo:
    def __init__(self, m, d, robot, meta):
        self.m, self.d, self.r = m, d, robot
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, J, n)]) for n in ("cart_x", "cart_y", "cart_yaw")]
        self.dadr = [int(m.jnt_dofadr[mujoco.mj_name2id(m, J, n)]) for n in ("cart_x", "cart_y", "cart_yaw")]
        self.body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cart")
        self.grasp = {}                 # side -> the grasp point on the handle (cart frame, x y z)
        self._tg_prev = None
        self.f = np.zeros(3)
        self.hold_s = 0.0
        self.travel = 0.0
        self._prev = None
        self.brake(True)

    def pose(self):
        return np.array([self.d.qpos[a] for a in self.qadr])

    def set_pose(self, x, y, yaw):
        for a, v in zip(self.qadr, (x, y, yaw)):
            self.d.qpos[a] = v
        for a in self.dadr:
            self.d.qvel[a] = 0.0

    def brake(self, on):
        damp = PH.CART_BRAKE_DAMPING if on else PH.CART_ROLL_DAMPING
        for a, v in zip(self.dadr, damp):
            self.m.dof_damping[a] = v

    def _site(self, side):
        return self.d.site_xpos[self.r.grasp_site[side]].copy()

    def _to_cart(self, p, pose=None):
        x, y, yaw = self.pose() if pose is None else pose
        c, s = math.cos(yaw), math.sin(yaw)
        d = np.asarray(p, float) - np.array([x, y, 0.0])
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])

    def engage(self, side, t):
        p = self._site(side)
        loc = self._to_cart(p)
        hz, hx = RM.CART["handle_z"], RM.CART["handle_x"]
        loc[0], loc[2] = hx, hz           # the grasp on the handle's axis (the grip rule put the hand there)
        self.grasp[side] = loc
        self._tg_prev = None
        self.brake(False)

    def release(self, side, t):
        self.grasp.pop(side, None)
        self._tg_prev = None
        if not self.grasp:
            self.brake(True)
            self.f[:] = 0.0
            for a in self.dadr:
                self.d.qfrc_applied[a] = 0.0

    def target(self):
        """The cart pose the holding hands imply (x, y, yaw), or None."""
        sides = sorted(self.grasp)
        if not sides:
            return None
        P = [self._site(s) for s in sides]
        L = [self.grasp[s] for s in sides]
        x, y, yaw = self.pose()
        if len(sides) == 2:
            dw = P[1][:2] - P[0][:2]
            dl = L[1][:2] - L[0][:2]
            yaw_t = math.atan2(dw[1], dw[0]) - math.atan2(dl[1], dl[0])
            yaw_t = yaw + wrap(yaw_t - yaw)
        else:
            yaw_t = yaw                    # one hand: the cart keeps its heading unless pulled round (below)
        c, s = math.cos(yaw_t), math.sin(yaw_t)
        Rt = np.array([[c, -s], [s, c]])
        pm = np.mean([p[:2] for p in P], axis=0)
        lm = np.mean([lc[:2] for lc in L], axis=0)
        xy = pm - Rt @ lm
        if len(sides) == 1:
            # a single hand drags the cart by its handle: the cart turns about the handle towards the pull
            v = np.array([x, y]) - P[0][:2]
            ang = math.atan2(v[1], v[0])
            lv = -L[0][:2]
            yaw_t = yaw + wrap((ang - math.atan2(lv[1], lv[0])) - yaw) * 0.5
            c, s = math.cos(yaw_t), math.sin(yaw_t)
            Rt = np.array([[c, -s], [s, c]])
            xy = P[0][:2] - Rt @ L[0][:2]
        return np.array([xy[0], xy[1], yaw_t])

    def pre_step(self):
        """Before the physics substeps: the servo's force (held) or nothing (braked)."""
        tg = self.target()
        if tg is None:
            return
        q = self.pose()
        qd = np.array([self.d.qvel[a] for a in self.dadr])
        e = tg - q
        e[2] = wrap(e[2])
        vt = np.zeros(3) if self._tg_prev is None else (tg - self._tg_prev) / TICK
        vt[2] = wrap(tg[2] - self._tg_prev[2]) / TICK if self._tg_prev is not None else 0.0
        self._tg_prev = tg.copy()
        vt = np.clip(vt, -1.5, 1.5)
        f = np.array(PH.CART_KP) * e + np.array(PH.CART_KD) * (vt - qd)
        k = len(self.grasp)
        fmax = PH.GRIP_MAX_FORCE * k * 1.6
        n = math.hypot(f[0], f[1])
        if n > fmax:
            f[:2] *= fmax / n
        tmax = PH.GRIP_MAX_TORQUE_BAR * k * 1.6
        f[2] = float(np.clip(f[2], -tmax, tmax))
        self.f = f
        for a, v in zip(self.dadr, f):
            self.d.qfrc_applied[a] = v

    def load(self, hands):
        """(load per hand N, over the grip's limit, the hands that left the handle)."""
        k = max(1, len(hands))
        share = math.hypot(self.f[0], self.f[1]) / k
        over = share > PH.GRIP_MAX_FORCE or abs(self.f[2]) / k > PH.GRIP_MAX_TORQUE_BAR
        lost = []
        q = self.pose()
        for s in hands:
            loc = self._to_cart(self._site(s), q)
            g = self.grasp.get(s)
            if g is None:
                continue
            if abs(loc[2] - g[2]) > PH.CART_HANDLE_OFF or abs(loc[0] - g[0]) > PH.CART_HANDLE_OFF or \
                    abs(loc[1] - g[1]) > PH.CART_HANDLE_OFF:
                lost.append(s)
        return share, over, lost

    def post_step(self):
        q = self.pose()
        if self._prev is not None:
            self.travel += math.hypot(q[0] - self._prev[0], q[1] - self._prev[1])
        self._prev = q
        if self.grasp:
            self.hold_s += TICK

    def corners(self, pose=None):
        x, y, yaw = self.pose() if pose is None else pose
        sx, sy = RM.CART["size"]
        c, s = math.cos(yaw), math.sin(yaw)
        hx = max(sx / 2, -RM.CART["handle_x"] + 0.02)
        pts = [(hx, sy / 2), (hx, -sy / 2), (-hx, sy / 2), (-hx, -sy / 2)]
        return [(x + c * a - s * b, y + s * a + c * b) for a, b in pts]


class Shift:
    """The service around the robot: the guests leaving, the spill, the dish crew, the parked bodies."""

    def __init__(self, m, d, meta, layout, grips, process, cart: CartServo):
        self.m, self.d = m, d
        self.meta, self.layout = meta, layout
        self.grips, self.process, self.cart = grips, process, cart
        B = mujoco.mjtObj.mjOBJ_BODY
        self.tables = dict(layout["tables"])
        self.guest_mocap = {}
        self.guest_geoms = {}
        for seat, gname in meta["guests"].items():
            b = mujoco.mj_name2id(m, B, gname)
            self.guest_mocap[seat] = int(m.body_mocapid[b])
            self.guest_geoms[seat] = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b]
        self.seat_table = {st["id"]: n for n, t in RM.tables().items() for st in t["seats"]}
        self.left_at = {n: (0.0 if s["state"] == "dirty" else None) for n, s in self.tables.items()}
        self.events = []
        self.parked = set(layout.get("parked", []))
        self.delivered = []                 # (t, name, kind)
        self.crew_visits = []
        self.dock_since = None
        self.dock_served = False
        self.free_adr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")])
                         for n in layout["loose"]}
        self.free_dof = {n: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_free")])
                         for n in layout["loose"]}
        self.kinds = {n: k for n, (k, _p) in layout["loose"].items()}
        self.park_slot = 0
        self.rollup_pool = [n for n in layout["rollups"] if n in self.parked]
        self.spill_started = False
        # what the score needs from during the shift: when the cone first stood beside the spill and when it left,
        # the seated parties' things a hand took
        self.monitor = {"cone_placed_at": None, "cone_left_at": None, "rude": []}
        self.cone_body = mujoco.mj_name2id(m, B, "cone") if "cone" in layout["loose"] else -1
        self.item_table = {}
        for n, (k, _p) in layout["loose"].items():
            if k in O.GUEST_KINDS:
                self.item_table[n] = self.seat_table.get(n.split("_", 1)[1])
        grips.on_engage.append(self._taken)

    def _taken(self, side, name, t):
        T = self.item_table.get(name)
        if T is not None and self.seated(T, t):
            self.monitor["rude"].append(name)
            self.events.append({"t": round(t, 2), "what": "took_from_seated", "table": T, "item": name})

    def cone_state(self):
        """(upright, held, xy) of the wet-floor cone now."""
        b = self.cone_body
        R = self.d.xmat[b].reshape(3, 3)
        held = "cone" in self.grips.held()
        return R[2, 2] > math.cos(math.radians(15.0)), held, self.d.xpos[b][:2].copy()

    def _watch_cone(self, t):
        sp = self.process.spill
        if not sp.active or self.cone_body < 0:
            return
        up, held, xy = self.cone_state()
        dist = float(np.linalg.norm(xy - sp.center))
        mon = self.monitor
        if mon["cone_placed_at"] is None:
            if up and not held and 0.30 <= dist <= 1.20:
                wet = sp.wet_cells()
                clear = True
                if wet.any():
                    dmin = float(np.min(np.linalg.norm(sp.xy[wet] - xy[None, :], axis=1)))
                    clear = dmin > 0.10 + 0.15
                if clear:
                    mon["cone_placed_at"] = round(t, 2)
                    self.events.append({"t": round(t, 2), "what": "cone_placed", "dist": round(dist, 3)})
        elif mon["cone_left_at"] is None:
            if held or not up or dist > 1.50:
                mon["cone_left_at"] = round(t, 2)
                self.events.append({"t": round(t, 2), "what": "cone_left"})

    def cone_home(self):
        if self.cone_body < 0:
            return False
        up, held, xy = self.cone_state()
        return up and not held and float(np.linalg.norm(xy - np.array(RM.CONE_HOME))) <= RM.CONE_HOME_TOL

    # ------------------------------------------------------------------------------------------ parking
    def park(self, name):
        a, dv = self.free_adr[name], self.free_dof[name]
        k = len(self.parked) + self.park_slot
        self.park_slot += 1
        x, y, z = PARK_AT[0] + 0.3 * (k % 30), PARK_AT[1] + 0.4 + 0.3 * (k // 30), 0.05
        self.d.qpos[a:a + 3] = (x, y, z)
        self.d.qpos[a + 3:a + 7] = (1.0, 0.0, 0.0, 0.0)
        self.d.qvel[dv:dv + 6] = 0.0
        self.parked.add(name)

    def unpark(self, name, pos, quat):
        a, dv = self.free_adr[name], self.free_dof[name]
        self.d.qpos[a:a + 3] = pos
        self.d.qpos[a + 3:a + 7] = quat
        self.d.qvel[dv:dv + 6] = 0.0
        self.parked.discard(name)

    # ------------------------------------------------------------------------------------------ events
    def guests_leave(self, table, t):
        for seat, mid in self.guest_mocap.items():
            if self.seat_table[seat] != table:
                continue
            self.d.mocap_pos[mid] = (PARK_AT[0] - 3.0 + 0.6 * len(self.events), PARK_AT[1] + 6.0, -2.0)
            for g in self.guest_geoms[seat]:
                self.m.geom_contype[g] = 0
                self.m.geom_conaffinity[g] = 0
        self.left_at[table] = t
        self.events.append({"t": round(t, 2), "what": "party_left", "table": table})

    def seated(self, table, t):
        s = self.tables[table]
        return s["state"] == "occupied" and self.left_at[table] is None

    def tick(self, t):
        for n, s in self.tables.items():
            if s["state"] == "occupied" and self.left_at[n] is None and t >= s["leave_s"]:
                self.guests_leave(n, t)
        if self.spill_started and int(round(t / TICK)) % 10 == 0:
            self._watch_cone(t)
        if not self.spill_started and t >= self.process.spill.t_spill:
            self.spill_started = True
            self.process.spill.start(t)
            self.events.append({"t": round(t, 2), "what": "spill", "table": self.layout["spill"]["table"],
                                "center": list(self.layout["spill"]["center"])})
        self._crew(t)

    # ------------------------------------------------------------------------------------------ the dish crew
    def cart_docked(self):
        if self.cart.grasp:
            return False
        x0, y0, x1, y1 = RM.DOCK["box"]
        x, y, yaw = self.cart.pose()
        if abs(wrap(2 * (yaw - RM.DOCK["yaw"])) / 2) > RM.DOCK["tol_yaw"]:
            return False
        return all(x0 <= px <= x1 and y0 <= py <= y1 for px, py in self.cart.corners())

    def _crew(self, t):
        if not self.cart_docked():
            self.dock_since = None
            self.dock_served = False
            return
        if self.dock_since is None:
            self.dock_since = t
        if self.dock_served or t - self.dock_since < RM.DOCK["crew_after_s"]:
            return
        self.dock_served = True
        rec = self.meta["cart"]
        x0, y0, z0, x1, y1, z1 = rec["tub_inside"]
        taken = []
        q = self.cart.pose()
        for n, kind in self.kinds.items():
            if n in self.parked or kind in ("cloth", "crumb_pan", "cone", "mop"):
                continue
            if n in self.grips.held():
                continue
            a = self.free_adr[n]
            loc = self.cart._to_cart(self.d.qpos[a:a + 3], q)
            if x0 - 0.01 < loc[0] < x1 + 0.01 and y0 - 0.01 < loc[1] < y1 + 0.01 and z0 - 0.02 < loc[2] < z1 + 0.10:
                taken.append(n)
        for n in taken:
            self.park(n)
            self.delivered.append((round(t, 2), n, self.kinds[n]))
        crumbs = self.process.crumbs.deliver_tub(PARK_AT)
        added = []
        # fresh sanitizer
        cl = self.process.cloth
        if self.process.in_bucket(self.d.xpos[self.process.cloth_body]):
            cl.charge, cl.load = 1.0, 0.0
        self.crew_visits.append({"t": round(t, 2), "took": len(taken), "crumbs": crumbs})
        self.events.append({"t": round(t, 2), "what": "crew", "took": len(taken)})
