"""The dining room's image light (cell internals): a procedural high-dynamic-range environment of the bistro as seen
from the middle of the room (the warm plaster ceiling with its dark beams and the glowing pendants, the cream walls
over the green wainscot, the bright street front with the sky and the sunlit far side of the street, the oak floor
with the sun's patches near the windows) turned into the cube map Filament lights the scene with, written as a KTX1
file the way Filament's cmgen writes one: prefiltered specular levels plus the irradiance's spherical harmonics.

The cube-map conventions and the KTX writer follow tasks/ground/earthquake_search_and_rescue/tests/rubblecell/sky.py
(MuJoCo's Filament samples a cube map in a y-up frame: a world direction (x, y, z) (z up) reads the cube at
(x, z, -y)). Everything is deterministic: the same bytes on every machine.
"""

from __future__ import annotations

import math
import struct

import numpy as np

__all__ = ["cube_dirs", "write_ktx", "irradiance_sh", "bistro_ibl"]

GL_R11F_G11F_B10F = 0x8C3A
GL_RGB = 0x1907


def cube_dirs(n):
    """(6, n, n, 3) unit directions of the texel centres of an OpenGL cube map (faces +x, -x, +y, -y, +z, -z; row 0 is
    the face's top), and (n, n) solid angles of one face's texels."""
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    sc, tc = np.meshgrid(c, c)
    one = np.ones_like(sc)
    faces = [np.stack([one, -tc, -sc], -1), np.stack([-one, -tc, sc], -1), np.stack([sc, one, tc], -1),
             np.stack([sc, -one, -tc], -1), np.stack([sc, -tc, one], -1), np.stack([-sc, -tc, -one], -1)]
    d = np.stack(faces, 0)
    l2 = (d * d).sum(-1, keepdims=True)
    d = d / np.sqrt(l2)
    dw = (2.0 / n) ** 2 / (l2[0, ..., 0] ** 1.5)
    return d, dw


def _pack_r11g11b10(rgb):
    x = np.clip(np.asarray(rgb, np.float32), 0.0, 64000.0)
    h = x.astype(np.float16).view(np.uint16).astype(np.uint32)
    r = (h[..., 0] >> 4) & 0x7FF
    g = (h[..., 1] >> 4) & 0x7FF
    b = (h[..., 2] >> 5) & 0x3FF
    return (r | (g << 11) | (b << 22)).astype("<u4")


def write_ktx(levels, sh=None):
    """KTX1 bytes of a cube map: `levels` = [(6, n, n, 3) float32, ...] from the largest; `sh` = (9, 3) or None."""
    n0 = levels[0].shape[1]
    kv = b""
    if sh is not None:
        txt = "".join("%g %g %g\n" % tuple(float(v) for v in row) for row in sh).encode()
        body = b"sh\x00" + txt
        kv = struct.pack("<I", len(body)) + body
        kv += b"\x00" * ((4 - len(kv) % 4) % 4)
    hdr = struct.pack("<13I", 0x04030201, GL_R11F_G11F_B10F, 1, GL_RGB, GL_R11F_G11F_B10F, GL_R11F_G11F_B10F,
                      n0, n0, 0, 0, 6, len(levels), len(kv))
    out = [b"\xabKTX 11\xbb\r\n\x1a\n", hdr, kv]
    for lv in levels:
        n = lv.shape[1]
        out.append(struct.pack("<I", n * n * 4))
        for f in range(6):
            out.append(_pack_r11g11b10(lv[f]).tobytes())
    return b"".join(out)


def _to_world(c):
    return np.stack([c[..., 0], -c[..., 2], c[..., 1]], -1)


def irradiance_sh(rad, dw, d):
    """Filament's shader-ready irradiance SH (9 x rgb) of a cube map of radiance (6, n, n, 3)."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    polys = [np.ones_like(x), y, z, x, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y]
    sp = math.sqrt(math.pi)
    K = [1 / (2 * sp), math.sqrt(3) / (2 * sp), math.sqrt(3) / (2 * sp), math.sqrt(3) / (2 * sp),
         math.sqrt(15) / (2 * sp), math.sqrt(15) / (2 * sp), math.sqrt(5) / (4 * sp), math.sqrt(15) / (2 * sp),
         math.sqrt(15) / (4 * sp)]
    Ahat = [math.pi, 2 * math.pi / 3, 2 * math.pi / 3, 2 * math.pi / 3] + [math.pi / 4] * 5
    w = dw[None, ..., None]
    out = np.zeros((9, 3))
    for i in range(9):
        proj = (rad * polys[i][..., None] * w).sum(axis=(0, 1, 2))
        out[i] = Ahat[i] * K[i] * K[i] * proj / math.pi
    return out


# ------------------------------------------------------------------------------------------------ the dining room
ROOM = {
    "probe": (5.2, 4.15, 1.4), "size": (10.4, 8.3), "ceiling_z": 3.2, "wainscot_z": 1.05,
    "window_x": ((0.40, 2.30), (2.50, 4.40), (4.60, 6.50), (6.70, 8.20)), "window_z": (0.45, 2.80),
    "ceiling_rgb": (0.62, 0.58, 0.52), "beam_rgb": (0.16, 0.11, 0.07), "plaster_rgb": (0.62, 0.56, 0.47),
    "wainscot_rgb": (0.07, 0.12, 0.09), "floor_rgb": (0.30, 0.20, 0.12), "sky_rgb": (2.0, 2.2, 2.6),
    "street_rgb": (1.9, 1.8, 1.6), "sunpatch_rgb": (2.4, 1.9, 1.3), "pendant_rgb": (14.0, 11.0, 7.5),
    "pendants": ((1.25, 0.90), (3.35, 0.90), (5.45, 0.90), (2.06, 4.095), (2.54, 4.095), (4.56, 4.095),
                 (5.04, 4.095), (1.25, 7.29), (3.35, 7.29), (5.45, 7.29)), "pendant_z": 2.1, "pendant_r": 0.08,
}


def _radiance(dirs, blur):
    """Radiance of the dining room seen from the probe along world directions (N, 3), smoothed by `blur`."""
    R = ROOM
    p = np.asarray(R["probe"], np.float64)
    d = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    X, Y = R["size"]
    out = np.zeros(d.shape, np.float64)
    # the first surface along each ray: the ceiling, the floor or a wall
    big = 1e9
    tz = np.where(d[:, 2] > 1e-6, (R["ceiling_z"] - p[2]) / np.maximum(d[:, 2], 1e-6),
                  np.where(d[:, 2] < -1e-6, (0.0 - p[2]) / np.minimum(d[:, 2], -1e-6), big))
    tx = np.where(d[:, 0] > 1e-6, (X - p[0]) / np.maximum(d[:, 0], 1e-6),
                  np.where(d[:, 0] < -1e-6, (0.0 - p[0]) / np.minimum(d[:, 0], -1e-6), big))
    ty = np.where(d[:, 1] > 1e-6, (Y - p[1]) / np.maximum(d[:, 1], 1e-6),
                  np.where(d[:, 1] < -1e-6, (0.0 - p[1]) / np.minimum(d[:, 1], -1e-6), big))
    t = np.minimum(np.minimum(tz, tx), ty)
    hit = p[None, :] + d * t[:, None]
    up = (tz <= tx) & (tz <= ty) & (d[:, 2] > 0)
    down = (tz <= tx) & (tz <= ty) & (d[:, 2] < 0)
    wall = ~(up | down)
    # ceiling: plaster with beams across x every 2.6 m
    beam = np.abs(np.mod(hit[:, 0] - 1.30 + 1.30, 2.60) - 1.30) < (0.09 + blur * 0.5)
    ceil = np.where(beam[:, None], np.asarray(R["beam_rgb"])[None], np.asarray(R["ceiling_rgb"])[None])
    out[up] = ceil[up]
    # floor: oak, warmer and brighter in the sun's patches near the street front
    fl = np.asarray(R["floor_rgb"])[None].repeat(len(d), 0) * (1.0 + 0.8 * np.exp(-hit[:, 1] / 1.2))[:, None]
    sunny = hit[:, 1] < 1.4
    fl[sunny] = fl[sunny] * 0.5 + np.asarray(R["sunpatch_rgb"])[None] * 0.5 * np.exp(-hit[sunny, 1] / 0.9)[:, None]
    out[down] = fl[down]
    # walls: cream plaster over the green wainscot; the south wall's windows (sky above, the street below)
    wl = np.where((hit[:, 2] < R["wainscot_z"])[:, None], np.asarray(R["wainscot_rgb"])[None],
                  np.asarray(R["plaster_rgb"])[None])
    south = wall & (ty <= tx) & (d[:, 1] < 0)
    inwin = np.zeros(len(d), bool)
    for a, b in R["window_x"]:
        inwin |= (hit[:, 0] > a) & (hit[:, 0] < b)
    inwin &= (hit[:, 2] > R["window_z"][0]) & (hit[:, 2] < R["window_z"][1])
    win = south & inwin
    sky = hit[:, 2] > 1.9
    wl[win & sky] = np.asarray(R["sky_rgb"])[None]
    wl[win & ~sky] = np.asarray(R["street_rgb"])[None]
    out[wall] = wl[wall]
    # the pendants' glowing shades (seen from the probe), spread by the blur
    for (x, y) in R["pendants"]:
        v = np.array([x, y, R["pendant_z"]]) - p
        dist = np.linalg.norm(v)
        v /= dist
        ang = np.arccos(np.clip(d @ v, -1, 1))
        r = math.atan(R["pendant_r"] / dist) + blur
        k = (math.atan(R["pendant_r"] / dist) / r) ** 2
        glow = np.clip(1.0 - ang / r, 0.0, 1.0) * k
        out += np.asarray(R["pendant_rgb"])[None] * glow[:, None]
    return out


def bistro_ibl(size=64, levels=None):
    """KTX1 bytes of the dining room's image light (prefiltered levels + irradiance SH), deterministic."""
    n = size
    nlev = levels or (int(math.log2(size)) + 1)
    out = []
    for i in range(nlev):
        d, _ = cube_dirs(n)
        tt = i / max(1, nlev - 1)
        pr = 1.0 - math.sqrt(max(0.0, 1.0 - tt))
        blur = max(2.0 / n, (pr * pr) * 1.1)
        rad = _radiance(_to_world(d).reshape(-1, 3), blur).reshape(6, n, n, 3)
        if blur > 0.05:
            k = max(1, int(round(blur * n / 2)))
            from scipy.ndimage import uniform_filter
            rad = np.stack([uniform_filter(rad[f], size=(2 * k + 1, 2 * k + 1, 1), mode="nearest") for f in range(6)])
        out.append(rad.astype(np.float32))
        n = max(1, n // 2)
    d, dw = cube_dirs(32)
    rad = _radiance(_to_world(d).reshape(-1, 3), 0.2).reshape(6, 32, 32, 3)
    sh = irradiance_sh(rad, dw, d)
    return write_ktx(out, sh)
