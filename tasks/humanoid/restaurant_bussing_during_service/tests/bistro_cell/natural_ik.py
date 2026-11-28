"""The reference's arm IK with a person's posture (ported from house_cleaning_with_tools's reference: NaturalIK).

The cell's damped least squares for the grasp site plus a posture term descended in the arm's null space (the
elbow down and a little out, the wrist near neutral, the joints off their limits), so planned arm configurations look
and move like a person's and stay away from contorted solutions that sweep the arm through furniture.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .robot import ARM_HOME, _quat_err


def _mirror(q):
    """A right-arm posture as the left arm's (the roll and yaw joints change sign)."""
    return [q[0], -q[1], -q[2], q[3], -q[4], q[5], -q[6]]


# natural reaching postures (right arm) that seed the IK when the straight approach line fails
_REACH_R = [[-0.30, -0.20, 0.30, 0.30, 0.80, 0.0, 0.0], [-0.30, 0.10, -0.20, -0.20, 1.50, 0.20, -0.80],
            [-0.60, -0.15, 0.0, 0.50, 0.0, 0.0, 0.0], [0.0, -0.30, 0.0, 0.80, -1.20, 0.0, 0.0],
            [-0.90, -0.20, 0.20, 0.90, 0.60, 0.0, 0.0]]
REACH_SEEDS = {"right": _REACH_R, "left": [_mirror(q) for q in _REACH_R]}


class NaturalIK:
    """The arm IK of the reference's plans: the cell's damped least squares (robot.ArmIK: the grasp site's position and
    rotation by the arm's 7 joints, the base, the waist and the legs as given), plus a posture term descended in the
    arm's null space, the redundancy the target leaves (the elbow's swivel about the shoulder-wrist line; with a
    position-only target the wrist as well). The term (terms) resolves the redundancy the way a person holds the arm:
    the swivel angle that puts the elbow down and a little out (the elbow on its circle about the shoulder-wrist line
    pulled towards that direction, always, not only past a threshold), the elbow not above the shoulder, not behind the
    body and not far out beside it, the wrist near its neutral (its sideways bend within 0.3 rad, its flexion within 0.55 rad, its roll within
    1 rad: the joint limits of the robot's wrist are far wider than a person's), the shoulder's twist within 1 rad, the
    elbow neither straightened past straight nor folded tight, every joint 0.35 rad off its limits. A target the arm
    can reach with the elbow up or down is so reached the way a person holds the arm, and posture_cost ranks grasps,
    stances and carries by the same measure. The posture step is capped per iteration, so a path solved waypoint by
    waypoint (each from the previous solution) drifts towards the natural posture smoothly instead of jumping."""

    LIMIT_MARGIN = 0.35        # rad: a joint closer than this to a limit costs
    ELBOW_RAISE = 0.02         # m: the elbow above the shoulder by more than this costs (reaching forward at shoulder
                               # height a person's elbow is level with the shoulder; above it the arm is raised)
    ELBOW_BACK = 0.05          # m: the elbow behind the shoulder (torso frame) beyond this costs
    ELBOW_OUT = 0.14           # m: the elbow out to the side of the shoulder beyond this costs
    # the elbow's natural direction about the shoulder-wrist line (torso frame: forward, out to the arm's side, up):
    # down and a little out and back; for a hand hanging straight below the shoulder, back (the forearm comes forward)
    SWIVEL_DIR = (-0.10, 0.35, -1.0)
    SWIVEL_DIR_HANG = (-1.0, 0.30, 0.0)
    # comfortable joint ranges (rad) beyond which a joint costs (a person's wrist and forearm, the robot's joints):
    # (joint index, half range, scale)
    COMFORT = ((5, 0.30, 0.25),     # wrist pitch: the hand bent sideways (radial / ulnar)
               (6, 0.55, 0.30),     # wrist yaw: the hand bent towards the palm or its back
               (4, 1.00, 0.40),     # wrist roll: the forearm turned
               (2, 1.00, 0.40))     # shoulder yaw: the upper arm twisted
    ELBOW_STRAIGHT = 1.45       # rad: the elbow joint past this is (nearly) straightened through (pi/2 is straight)
    ELBOW_FOLD = -0.70          # rad: below this the elbow is folded tight (0: a right angle)
    # (the swivel dominates the comfort terms: with the hand's pose given, a lower elbow often bends the wrist more, and
    # a raised or winged elbow reads as far more contorted than a bent wrist; the joint limits dominate both, so the
    # swivel never wrings a wrist against its stop)
    W = {"limit": 30.0, "elbow_high": 3.0, "elbow_back": 1.0, "elbow_out": 0.5, "swivel": 8.0, "comfort": 0.6,
         "elbow_straight": 1.5, "elbow_fold": 0.4, "twist": 0.04}
    # the ranking measure's constants (terms_rank)
    RANK_ELBOW_DROP, RANK_ELBOW_BACK, RANK_ELBOW_OUT = 0.08, 0.05, 0.12
    RANK_W = {"limit": 1.0, "elbow_high": 3.0, "elbow_back": 1.0, "elbow_out": 0.5, "twist": 0.08}

    def __init__(self, m, r):
        self.m, self.r = m, r
        self.dk = mujoco.MjData(m)
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))
        self.je = np.zeros((3, m.nv))
        B = mujoco.mjtObj.mjOBJ_BODY
        self.elbow = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_elbow_link") for s in ("left", "right")}
        self.shoulder = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_shoulder_pitch_link") for s in ("left", "right")}
        self.wrist = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_wrist_pitch_link") for s in ("left", "right")}
        self.wroll = {s: mujoco.mj_name2id(m, B, f"{r.name}/{s}_wrist_roll_link") for s in ("left", "right")}
        self.out = {"left": 1.0, "right": -1.0}

    def natural_elbow(self, side, Rt):
        """The elbow's natural place (torso frame, relative to the shoulder) on its circle about the shoulder-wrist
        line at the present wrist, and the circle's radius (m): (point, radius)."""
        dk = self.dk
        S = dk.xpos[self.shoulder[side]]
        E = Rt.T @ (dk.xpos[self.elbow[side]] - S)
        Wp = Rt.T @ (dk.xpos[self.wrist[side]] - S)
        d = float(np.linalg.norm(Wp))
        if d < 1e-6:
            return E, 0.0
        n = Wp / d
        a, b = float(np.linalg.norm(E)), float(np.linalg.norm(Wp - E))
        x = (a * a - b * b + d * d) / (2.0 * d)
        rho = math.sqrt(max(0.0, a * a - x * x))
        o = self.out[side]
        want = np.array([self.SWIVEL_DIR[0], o * self.SWIVEL_DIR[1], self.SWIVEL_DIR[2]])
        hang = np.array([self.SWIVEL_DIR_HANG[0], o * self.SWIVEL_DIR_HANG[1], self.SWIVEL_DIR_HANG[2]])
        w = want - (want @ n) * n
        w = w + 0.3 * (hang - (hang @ n) * n)
        nw = float(np.linalg.norm(w))
        if nw < 1e-6:
            return E, 0.0
        return x * n + rho * (w / nw), rho

    def terms(self, side, q):
        """The posture term at q (self.dk's kinematics and comPos set) as weighted residuals: (residuals, their
        Jacobian over the arm's 7 joints, {term: cost}); the cost is the sum of the squared residuals."""
        r, dk, m = self.r, self.dk, self.m
        lo, hi = r.arm_lo[side], r.arm_hi[side]
        mu = self.LIMIT_MARGIN
        res, rows_, parts = [], [], {}
        wl = math.sqrt(self.W["limit"]) / mu
        c = 0.0
        for j in range(7):
            a, b = mu - (q[j] - lo[j]), mu - (hi[j] - q[j])
            for v_, sg in ((a, -1.0), (b, 1.0)):
                if v_ > 0:
                    jr = np.zeros(7)
                    jr[j] = sg * wl
                    res.append(wl * v_)
                    rows_.append(jr)
                    c += (wl * v_) ** 2
        parts["limit"] = c
        Rt = dk.xmat[r.torso].reshape(3, 3)
        v = Rt.T @ (dk.xpos[self.elbow[side]] - dk.xpos[self.shoulder[side]])
        mujoco.mj_jacBody(m, dk, self.je, None, self.elbow[side])
        Jv = Rt.T @ self.je[:, r.arm_v[side]]
        for name, e, jrow, scale in (("elbow_high", v[2] - self.ELBOW_RAISE, Jv[2], 0.08),
                                     ("elbow_back", -v[0] - self.ELBOW_BACK, -Jv[0], 0.1),
                                     ("elbow_out", self.out[side] * v[1] - self.ELBOW_OUT, self.out[side] * Jv[1],
                                      0.1)):
            w = math.sqrt(self.W[name]) / scale
            parts[name] = (w * e) ** 2 if e > 0 else 0.0
            if e > 0:
                res.append(w * e)
                rows_.append(w * jrow)
        # the swivel: the elbow pulled towards its natural place on its circle (faded out as the arm straightens,
        # where the swivel no longer moves the elbow)
        Ep, rho = self.natural_elbow(side, Rt)
        fade = min(1.0, rho / 0.06)
        ws = math.sqrt(self.W["swivel"]) / 0.10 * fade
        dv = v - Ep
        parts["swivel"] = float(ws * ws * (dv @ dv))
        if ws > 0:
            for k in range(3):
                res.append(ws * dv[k])
                rows_.append(ws * Jv[k])
        cc = 0.0
        for j, half, scale in self.COMFORT:
            e = abs(q[j]) - half
            if e > 0:
                w = math.sqrt(self.W["comfort"]) / scale
                jr = np.zeros(7)
                jr[j] = w * (1.0 if q[j] > 0 else -1.0)
                res.append(w * e)
                rows_.append(jr)
                cc += (w * e) ** 2
        parts["comfort"] = cc
        for name, e, sg, scale in (("elbow_straight", q[3] - self.ELBOW_STRAIGHT, 1.0, 0.15),
                                   ("elbow_fold", self.ELBOW_FOLD - q[3], -1.0, 0.30)):
            w = math.sqrt(self.W[name]) / scale
            parts[name] = (w * e) ** 2 if e > 0 else 0.0
            if e > 0:
                jr = np.zeros(7)
                jr[3] = w * sg
                res.append(w * e)
                rows_.append(jr)
        wt = math.sqrt(self.W["twist"])
        parts["twist"] = wt * wt * float(q[2] ** 2 + q[4] ** 2 + q[6] ** 2)
        for j in (2, 4, 6):
            jr = np.zeros(7)
            jr[j] = wt
            res.append(wt * q[j])
            rows_.append(jr)
        return np.array(res), np.array(rows_).reshape(-1, 7), parts

    def terms_rank(self, side, q):
        """The ranking measure of a posture (posture_cost): the joints off their limits, the elbow at least 8 cm below
        the shoulder, not behind it or far out beside it, the twisting joints near their middle (residuals, Jacobian,
        {term: cost}). The planner's choices between grasps, carries and stances rank by it, the solver's redundancy
        resolution (terms) is finer."""
        r, dk, m = self.r, self.dk, self.m
        lo, hi = r.arm_lo[side], r.arm_hi[side]
        mu = self.LIMIT_MARGIN
        res, rows_, parts = [], [], {}
        wl = math.sqrt(self.RANK_W["limit"]) / mu
        c = 0.0
        for j in range(7):
            a, b = mu - (q[j] - lo[j]), mu - (hi[j] - q[j])
            for v_, sg in ((a, -1.0), (b, 1.0)):
                if v_ > 0:
                    jr = np.zeros(7)
                    jr[j] = sg * wl
                    res.append(wl * v_)
                    rows_.append(jr)
                    c += (wl * v_) ** 2
        parts["limit"] = c
        Rt = dk.xmat[r.torso].reshape(3, 3)
        v = Rt.T @ (dk.xpos[self.elbow[side]] - dk.xpos[self.shoulder[side]])
        mujoco.mj_jacBody(m, dk, self.je, None, self.elbow[side])
        Jv = Rt.T @ self.je[:, r.arm_v[side]]
        for name, e, jrow, scale in (("elbow_high", v[2] + self.RANK_ELBOW_DROP, Jv[2], self.RANK_ELBOW_DROP),
                                     ("elbow_back", -v[0] - self.RANK_ELBOW_BACK, -Jv[0], 0.1),
                                     ("elbow_out", self.out[side] * v[1] - self.RANK_ELBOW_OUT, self.out[side] * Jv[1],
                                      0.1)):
            w = math.sqrt(self.RANK_W[name]) / scale
            parts[name] = (w * e) ** 2 if e > 0 else 0.0
            if e > 0:
                res.append(w * e)
                rows_.append(w * jrow)
        wt = math.sqrt(self.RANK_W["twist"])
        parts["twist"] = wt * wt * float(q[2] ** 2 + q[4] ** 2)
        for j in (2, 4):
            jr = np.zeros(7)
            jr[j] = wt
            res.append(wt * q[j])
            rows_.append(jr)
        return np.array(res), np.array(rows_).reshape(-1, 7), parts

    # the human posture measure (round 5: posture_cost, by which every choice of the planner ranks): what a person's
    # arm does not do, as squared hinges past a comfortable range, the quantities of the review's posture measure
    # (tests/review/posture.md): (joint, half range, scale) of the wrist (sideways bend, flexion, the forearm's turn)
    H_WRIST = ((5, 0.30, 0.22), (6, 0.45, 0.22), (4, 1.05, 0.25))
    H_TWIST = (1.00, 0.25)              # rad: the shoulder's twist
    H_ELBOW_DROP = 0.09                 # m: the elbow at least this far below the shoulder (torso frame)
    H_SWIVEL = (30.0, 15.0)             # deg: the elbow off its natural direction about the shoulder-wrist line
    H_LIMIT = 0.25                      # rad: every joint this far off its limits

    def human(self, side, q):
        """{term: cost} of arm joints q (self.dk's kinematics set; 0 for a posture a person takes without effort, about
        1 at the edge of the comfortable range, 4 and more where the review's measure flags it)."""
        r, dk = self.r, self.dk
        q = np.asarray(q, float)
        parts = {}
        lo, hi = r.arm_lo[side], r.arm_hi[side]
        marg = np.minimum(q - lo, hi - q)
        parts["limit"] = float(np.sum((np.maximum(0.0, self.H_LIMIT - marg) / 0.10) ** 2))
        parts["wrist"] = float(sum((max(0.0, abs(q[j]) - h) / s) ** 2 for j, h, s in self.H_WRIST))
        parts["twist"] = (max(0.0, abs(q[2]) - self.H_TWIST[0]) / self.H_TWIST[1]) ** 2
        Rt = dk.xmat[r.torso].reshape(3, 3)
        S = Rt.T @ dk.xpos[self.shoulder[side]]
        E = Rt.T @ dk.xpos[self.elbow[side]]
        Wr = Rt.T @ dk.xpos[self.wroll[side]]
        v = E - S
        o = self.out[side]
        parts["elbow_high"] = (max(0.0, v[2] + self.H_ELBOW_DROP) / 0.04) ** 2
        parts["elbow_out"] = (max(0.0, o * v[1] - 0.13) / 0.05) ** 2
        parts["elbow_back"] = (max(0.0, -v[0] - 0.06) / 0.05) ** 2
        n = Wr - S
        dn = float(np.linalg.norm(n))
        sw = 0.0
        if dn > 1e-6:
            n = n / dn
            e = v - (v @ n) * n
            want = np.array([0.0, 0.3 * o, -1.0])
            w = want - (want @ n) * n
            if np.linalg.norm(w) < 0.2:
                want = np.array([0.0, o, -0.3])
                w = want - (want @ n) * n
            a_, b_ = float(np.linalg.norm(v)), float(np.linalg.norm(Wr - E))
            bend = 180.0 - math.degrees(math.acos(float(np.clip((a_ * a_ + b_ * b_ - dn * dn) / (2 * a_ * b_), -1, 1))))
            if bend > 20.0 and np.linalg.norm(e) > 1e-6:
                dev = math.degrees(math.acos(float(np.clip((e / np.linalg.norm(e)) @ (w / np.linalg.norm(w)), -1, 1))))
                sw = (max(0.0, dev - self.H_SWIVEL[0]) / self.H_SWIVEL[1]) ** 2 * min(1.0, (bend - 20.0) / 15.0)
        parts["swivel"] = sw
        parts["hyperext"] = (max(0.0, q[3] - 1.50) / 0.08) ** 2
        return parts

    def cost(self, qpos, side, q):
        """The posture cost of arm joints q (the rest of the robot as in qpos): the human measure (human)."""
        dk = self.dk
        dk.qpos[:] = qpos
        dk.qpos[self.r.arm_q[side]] = q
        mujoco.mj_kinematics(self.m, dk)
        return sum(self.human(side, np.asarray(q, float)).values())

    def cost_rank(self, qpos, side, q):
        """The round-3 ranking measure (terms_rank) of arm joints q."""
        dk = self.dk
        dk.qpos[:] = qpos
        dk.qpos[self.r.arm_q[side]] = q
        mujoco.mj_kinematics(self.m, dk)
        mujoco.mj_comPos(self.m, dk)
        return sum(self.terms_rank(side, np.asarray(q, float))[2].values())

    def solve(self, qpos, side, pos, quat, q_init, iters=30, damping=0.05, w_rot=0.35, null_step=0.04, null_total=None,
              w_home=None, natural=False):
        """Arm joints putting `side`'s grasp site at pos (world) with orientation quat (wxyz; None: position only),
        the posture term descended in the null space (at most null_step rad per iteration and, if given, null_total
        rad in all: a path's waypoint so changes the arm's posture only gradually). Returns (q, pos_err_m,
        rot_err_rad). (w_home: ignored; the posture term replaces the cell IK's pull towards the home posture.)"""
        m, r, dk = self.m, self.r, self.dk
        dk.qpos[:] = qpos
        q = np.array(q_init, dtype=float)
        lo, hi = r.arm_lo[side] + 1e-3, r.arm_hi[side] - 1e-3
        q = np.clip(q, lo, hi)
        sid = r.grasp_site[side]
        pos = np.asarray(pos, dtype=float)
        quat = None if quat is None else np.asarray(quat, dtype=float) / np.linalg.norm(quat)
        ep, er = np.zeros(3), np.zeros(3)
        budget = math.inf if null_total is None else float(null_total)
        for it in range(iters + 1):
            dk.qpos[r.arm_q[side]] = q
            mujoco.mj_kinematics(m, dk)
            ep = pos - dk.site_xpos[sid]
            if quat is not None:
                cur = np.zeros(4)
                mujoco.mju_mat2Quat(cur, dk.site_xmat[sid])
                er = _quat_err(quat, cur)
            if it == iters:
                break
            mujoco.mj_comPos(m, dk)
            mujoco.mj_jacSite(m, dk, self.jp, self.jr, sid)
            Jp = self.jp[:, r.arm_v[side]]
            if quat is not None:
                J0 = np.vstack([Jp, w_rot * self.jr[:, r.arm_v[side]]])
                e = np.concatenate([ep, w_rot * er])
            else:
                J0, e = Jp, ep
            # (the posture term only once the target is nearly reached: far from it the step is the cell IK's own,
            # so whatever the cell IK reaches this reaches as well)
            near = np.linalg.norm(ep) < 0.02 and np.linalg.norm(er) < 0.2
            r2, J2, _ = (self.terms if natural else self.terms_rank)(side, q)
            free = np.ones(7, bool)
            for _pass in range(3):
                # a joint at its limit that the step would push further is held (its column dropped) and the step
                # solved again with the others
                J = J0 * free
                JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
                Jpinv = J.T @ np.linalg.solve(JJt, np.eye(J.shape[0]))
                d1 = Jpinv @ e
                # the posture residuals as a second task in the target's null space (task priority, damped), through
                # the exact projector onto that null space (a damped one would let the posture step disturb the
                # target, and the two then settle millimetres off it)
                _, sv, Vt = np.linalg.svd(J)
                Vr = Vt[:int(np.sum(sv > 1e-3))]
                N = (np.eye(7) - Vr.T @ Vr) * free[:, None] * free[None, :]
                J2N = J2 @ N
                dn = N @ (J2N.T @ np.linalg.solve(J2N @ J2N.T + 0.1 ** 2 * np.eye(len(r2)), -r2 - J2 @ d1)) if near \
                    else np.zeros(7)
                sn = float(np.max(np.abs(dn))) if near else 1.0
                cap = min(null_step, budget)
                if near and sn > cap:
                    dn *= cap / sn if sn > 0 else 0.0
                dq = d1 + dn
                step = np.max(np.abs(dq))
                if step > 0.35:
                    dq *= 0.35 / step
                stuck = free & (((q <= lo + 1e-6) & (dq < 0)) | ((q >= hi - 1e-6) & (dq > 0)))
                if not stuck.any():
                    break
                free &= ~stuck
            if near:
                budget -= min(sn, cap)
            if near and np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3 and (sn < 1e-3 or budget <= 0.0):
                break
            q = np.clip(q + dq, lo, hi)
        return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))


