"""The dining room of the bistro "Linden": fixed geometry in metres (the survey the cell manual publishes exactly).

Room frame: origin at the south-west floor corner, x east, y north, z up. The street front (big windows and the
entrance door) is the south wall, the kitchen wall (the kitchen door and the dish-return pass) the north wall, the
service station the east wall. Eight tables stand in three rows along x: three two-tops by the windows (row S), two
four-tops in the middle (row M) and three two-tops at the back (row N); every chair stands north or south of its table
(the guests face north or south), so a table is worked from its west and east ends, which the cross gaps between the
tables leave free. Two aisles run east-west between the rows; the east zone (x > 6.2) holds the service station, the
bus cart's home, the mop bucket and the wet-floor cone, the dish-return pass and its cart dock, the host stand and the
entrance. Boxes are (x0, y0, z0, x1, y1, z1).
"""

from __future__ import annotations

import math

ROOM_X, ROOM_Y, WALL_H = 10.4, 8.3, 3.2
WALL_T = 0.15

# ---------------------------------------------------------------------------------- openings
# the street front: four window bays (sill 0.45 m, head 2.80 m) and the glazed entrance door
WINDOWS = ((0.40, 2.30), (2.50, 4.40), (4.60, 6.50), (6.70, 8.20))
WINDOW_Z = (0.45, 2.80)
ENTRANCE_X, ENTRANCE_H = (8.50, 9.50), 2.45
KITCHEN_DOOR_X, KITCHEN_DOOR_H = (6.40, 7.20), 2.10        # north wall, a swing door (drawn closed)

# ---------------------------------------------------------------------------------- tables and chairs
TABLE_TOP_Z, TABLE_TOP_T = 0.75, 0.035
TWO_TOP = (0.60, 0.60)                    # x, y
FOUR_TOP = (1.00, 0.75)
CHAIR = {"seat": (0.42, 0.42), "seat_z": 0.46, "seat_t": 0.03, "back_h": 0.90, "back_t": 0.025, "leg": 0.03,
         "offset": 0.24}                 # the seat's centre this far outside the table's edge
PEDESTAL = {"column_r": 0.040, "base_half": 0.24, "base_h": 0.035}   # two-tops: a cast-iron cross base
LEG = {"side": 0.05, "inset": 0.07}                                     # four-tops: four legs
ROLLUP_FROM_EDGE = (0.05, 0.14)          # a roll-up's centre this far in from the table's edge in front of the seat

TABLE_CENTERS = {
    "T1": (1.25, 0.90), "T2": (3.35, 0.90), "T3": (5.45, 0.90),        # row S (windows)
    "T4": (2.30, 4.095), "T5": (4.80, 4.095),                          # row M (four-tops)
    "T6": (1.25, 7.29), "T7": (3.35, 7.29), "T8": (5.45, 7.29),        # row N (back)
}
TABLE_SEATS = {"T1": 2, "T2": 2, "T3": 2, "T4": 4, "T5": 4, "T6": 2, "T7": 2, "T8": 2}


def tables() -> dict:
    """name -> {center, size, top_z, seats: [{chair (x, y), facing (yaw the guest faces), side (S or N),
    setting (x, y: the cover's middle on the top), rollup (x, y, axis yaw): the reset roll-up's spot}], kind}."""
    out = {}
    for name, (cx, cy) in TABLE_CENTERS.items():
        n = TABLE_SEATS[name]
        sx, sy = TWO_TOP if n == 2 else FOUR_TOP
        xs = (cx,) if n == 2 else (cx - 0.24, cx + 0.24)
        seats = []
        for side, sgn in (("S", -1.0), ("N", 1.0)):
            edge = cy + sgn * sy / 2
            for x in xs:
                seats.append({"id": f"{name}{side}{'' if n == 2 else ('w' if x < cx else 'e')}",
                              "chair": (x, edge + sgn * CHAIR["offset"]),
                              "facing": -sgn * math.pi / 2,          # the guest faces the table: north for S seats
                              "side": side,
                              "setting": (x, edge - sgn * 0.15),
                              "rollup": (x, edge - sgn * 0.09, math.pi / 2)})
        out[name] = {"center": (cx, cy), "size": (sx, sy), "top_z": TABLE_TOP_Z, "seats": seats,
                     "kind": "two_top" if n == 2 else "four_top", "row": name_row(name)}
    return out


def name_row(name: str) -> str:
    return {"T1": "S", "T2": "S", "T3": "S", "T4": "M", "T5": "M"}.get(name, "N")


def table_box(name: str):
    t = tables()[name]
    (cx, cy), (sx, sy) = t["center"], t["size"]
    return (cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2)


# ---------------------------------------------------------------------------------- the east zone
# the service station (side stand) against the east wall: a cabinet with a wooden top, open shelves above it
STATION = {"box": (9.80, 2.70, 0.0, ROOM_X, 4.90, 0.92), "shelves_z": (1.35, 1.75), "shelf_depth": 0.30}
# the dish-return pass on the north wall: a stainless counter under an opening into the dish room
PASS = {"counter": (7.40, 7.78, 0.0, 9.40, ROOM_Y, 0.92), "opening_z": (0.92, 1.95)}
# the cart dock in front of the pass: a yellow taped rectangle on the floor; a cart left standing in it (its footprint
# inside, its long side towards the pass) is emptied by the dish crew
DOCK = {"box": (7.50, 6.95, 9.30, 7.74), "yaw": 0.0, "tol_yaw": math.radians(12.0), "crew_after_s": 20.0}
# the host stand by the entrance (a lectern) and the planters (fixed)
HOST_STAND = {"box": (7.45, 0.55, 0.0, 7.95, 0.95, 1.12)}
PLANTERS = {"nw": {"center": (0.40, 7.95), "radius": 0.22, "h": 0.55, "foliage_r": 0.42, "foliage_z": (0.55, 2.05)},
            "sw_window": {"center": (8.85, 0.35), "radius": 0.20, "h": 0.45, "foliage_r": 0.40,
                          "foliage_z": (0.45, 1.70)}}

# ---------------------------------------------------------------------------------- the bus cart, the mop, the cone
# the cart's frame: origin on the floor under the middle of its top shelf, x towards its front (away from the push
# handle), y to its left, z up
CART = {"size": (0.95, 0.40), "top_z": 0.76, "lower_z": 0.30, "lip": 0.022, "handle_z": 0.98, "handle_x": -0.545,
        "handle_half": 0.17, "handle_r": 0.0125, "caster_r": 0.050, "mass": 15.0,
        "home": (8.05, 3.80, math.pi),               # (x, y, yaw): its front to the west, by the station
        # what stands on its top shelf (cart frame): the bus tub at the front, the sanitizer bucket at the back (both
        # on its middle line); the crumb pan hangs on a hook at the back, below the push handle
        "tub_at": (0.180, 0.0), "bucket_at": (-0.300, 0.0),
        "pan_hook": (-0.505, 0.0, 0.74)}
TUB = {"outer": (0.58, 0.36), "h": 0.14, "wall_t": 0.006, "mass": 1.1}      # a shallow bus box (5.5 in)
BUCKET = {"r": 0.115, "h": 0.10, "fill_h": 0.088, "wall_t": 0.005, "rack_h": 0.070}   # a perforated rack inside:
# the cloth soaks on it just below the solution's surface
# the clean roll-ups: two wire baskets on the service station's counter (roll-ups lying along y, two layers of four)
BASKETS = ({"center": (9.93, 3.15), "size": (0.20, 0.30), "h": 0.07}, {"center": (9.93, 3.55), "size": (0.20, 0.30),
                                                                        "h": 0.07})
MOP_HOME = {"bucket": (10.15, 5.98), "bucket_size": (0.40, 0.28, 0.42),   # the mop bucket (its wringer box), by
            "clip": (10.30, 5.45, 1.10)}                                  # the mop; its handle clip on the east wall
CONE_HOME = (10.05, 6.58)                 # the wet-floor cone's spot (a taped square by the wall)
CONE_HOME_TOL = 0.25

# ---------------------------------------------------------------------------------- the robot and the cameras
ROBOT_HOME = (9.05, 2.95, math.pi)        # by the cart's handle, facing west (before the per-episode jitter)

CEILING_CAMERAS = {
    "ceiling_sw": {"pos": (0.25, 0.25, 2.90), "target": (3.20, 3.40, 0.30), "fovy": 72.0, "size": (960, 720)},
    "ceiling_nw": {"pos": (0.25, 8.05, 2.90), "target": (3.20, 4.90, 0.30), "fovy": 72.0, "size": (960, 720)},
    "ceiling_center": {"pos": (6.55, 4.15, 2.95), "target": (3.00, 4.15, 0.20), "fovy": 72.0, "size": (960, 720)},
    "ceiling_service": {"pos": (10.15, 0.25, 2.90), "target": (8.00, 4.90, 0.40), "fovy": 72.0, "size": (960, 720)},
}


def surfaces() -> dict:
    """Horizontal surfaces the crumbs come to rest on (highest wins): name -> (footprint box, height)."""
    out = {}
    for name, t in tables().items():
        (cx, cy), (sx, sy) = t["center"], t["size"]
        out[f"{name}_top"] = ((cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2), TABLE_TOP_Z)
        for s in t["seats"]:
            x, y = s["chair"]
            w, l = CHAIR["seat"]
            out[f"chair_{s['id']}"] = ((x - w / 2, y - l / 2, x + w / 2, y + l / 2), CHAIR["seat_z"])
    b = STATION["box"]
    out["station"] = ((b[0], b[1], b[3], b[4]), b[5])
    b = PASS["counter"]
    out["pass"] = ((b[0], b[1], b[3], b[4]), b[5])
    return out
