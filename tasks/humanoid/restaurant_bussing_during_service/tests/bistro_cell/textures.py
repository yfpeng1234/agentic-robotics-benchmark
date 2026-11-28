"""Procedural textures of the bistro cell (cell internals): uint8 RGB images (rows top-down) drawn with numpy and
Pillow from fixed seeds, the same pixels on every machine. Fine high-contrast detail is avoided on purpose: the
renderer samples textures without mipmaps, so a busy pattern would shimmer at a distance."""

from __future__ import annotations

import functools
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


@functools.lru_cache(maxsize=None)
def font(size, bold=False, serif=False, mono=False):
    name = "DejaVuSansMono" if mono else ("DejaVuSerif" if serif else "DejaVuSans")
    if bold:
        name += "-Bold"
    try:
        return ImageFont.truetype(f"{FONT_DIR}/{name}.ttf", int(size))
    except OSError:
        return ImageFont.load_default()


def _u8(img):
    return np.clip(np.round(np.asarray(img, np.float64) * 255.0), 0, 255).astype(np.uint8)


def noise(h, w, cells, seed, octaves=4, persistence=0.5):
    """Tileable value noise in [0, 1] (smooth bilinear, wrapped)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w))
    amp, tot = 1.0, 0.0
    cy, cx = cells
    for _ in range(octaves):
        g = rng.random((cy, cx))
        y = np.linspace(0, cy, h, endpoint=False)
        x = np.linspace(0, cx, w, endpoint=False)
        y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
        fy, fx = (y - y0)[:, None], (x - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        y1, x1 = (y0 + 1) % cy, (x0 + 1) % cx
        a = g[y0][:, x0] * (1 - fx) + g[y0][:, x1] * fx
        b = g[y1][:, x0] * (1 - fx) + g[y1][:, x1] * fx
        out += amp * (a * (1 - fy) + b * fy)
        tot += amp
        amp *= persistence
        cy, cx = cy * 2, cx * 2
    return out / tot


def _col(c):
    return np.asarray(c, np.float64)[None, None, :]


# ================================================================================================ surfaces
def wood(h, w, light, dark, seed, grain=30.0, warp=5.0):
    """Wood grain along the image's x axis (u)."""
    n1 = noise(h, w, (3, 2), seed, 4)
    n2 = noise(h, w, (24, 2), seed + 1, 3)
    y = np.linspace(0, 1, h, endpoint=False)[:, None]
    rings = (np.sin((y * grain + warp * n1) * 2 * math.pi) * 0.5 + 0.5) ** 2.0
    t = np.clip(0.50 * rings + 0.40 * n2 + 0.10 * noise(h, w, (64, 8), seed + 2, 2), 0, 1)
    img = _col(light) * (1 - t[..., None]) + _col(dark) * t[..., None]
    return _u8(img)


def chevron(size=1024, span=1.2, col_w=0.30, plank_w=0.085, seed=7):
    """Oak chevron parquet (point de Hongrie): `span` metres per texture, columns `col_w` wide whose planks lean +45
    and -45 degrees in turn (they meet in a point at the column joints), each plank with its own tint and grain, thin
    dark joints."""
    px = size / span
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
    x, y = xx / px, yy / px
    col = np.floor(x / col_w)
    xl = x - col * col_w
    sgn = np.where(col % 2 == 0, 1.0, -1.0)
    # (each column's planks: parallelograms whose long edges lean 45 deg; the plank index along y)
    k = np.floor((y + sgn * xl) / plank_w)
    pid = ((col.astype(np.int64) * 7919 + k.astype(np.int64) * 104729) % 9973).astype(int)
    rng = np.random.default_rng(seed)
    tint = rng.uniform(0.86, 1.10, size=9973)[pid]
    hue = rng.uniform(-0.04, 0.04, size=9973)[pid]
    g = noise(size, size, (12, 12), seed + 3, 3)
    along = (xl - sgn * y) / math.sqrt(2.0)          # along the plank (its grain)
    across = (y + sgn * xl) / math.sqrt(2.0)
    grain = 0.5 + 0.5 * np.sin(across * 260.0 + 6.0 * np.sin(along * 9.0 + pid) + g * 6.0)
    base_light, base_dark = np.array([0.70, 0.56, 0.41]), np.array([0.50, 0.38, 0.26])
    t = np.clip(0.30 * grain ** 2 + 0.50 * g + 0.20 * noise(size, size, (96, 96), seed + 4, 1), 0, 1)
    img = base_light[None, None, :] * (1 - t[..., None]) + base_dark[None, None, :] * t[..., None]
    img = img * tint[..., None]
    img[..., 0] *= (1 + hue)
    img[..., 2] *= (1 - hue)
    joint = np.zeros((size, size), bool)
    for arr in (col, k):
        joint[:, 1:] |= arr[:, 1:] != arr[:, :-1]
        joint[1:, :] |= arr[1:, :] != arr[:-1, :]
    img[joint] *= 0.60
    # a little wear: lighter where people walk (low-frequency)
    img *= (0.97 + 0.06 * noise(size, size, (3, 3), seed + 5, 2))[..., None]
    return _u8(np.clip(img, 0, 1))


def plaster(h, w, base, seed):
    n = noise(h, w, (6, 6), seed, 5)
    m = noise(h, w, (30, 30), seed + 1, 2)
    return _u8(_col(base) * (0.93 + 0.07 * n[..., None] + 0.025 * m[..., None]))


def wainscot(h, w, paint, seed, rails=((0.0, 0.08), (0.90, 1.0)), stiles=0.07):
    """A painted panelled wainscot (one module: the texture spans one panel's width and the wainscot's height): a
    raised frame round a recessed panel, lit from above (light top edges, dark bottom edges)."""
    img = np.ones((h, w, 3)) * _col(paint)
    img *= (0.96 + 0.05 * noise(h, w, (4, 4), seed, 3))[..., None]
    y = np.arange(h)[:, None] / h
    x = np.arange(w)[None, :] / w
    inner = (y > 0.16) & (y < 0.84) & (x > stiles) & (x < 1 - stiles)
    img[inner] *= 0.90
    edge = 3
    yi0, yi1 = int(0.16 * h), int(0.84 * h)
    xi0, xi1 = int(stiles * w), int((1 - stiles) * w)
    img[yi0:yi0 + edge, xi0:xi1] *= 0.70            # shadow under the frame's top rail
    img[yi1 - edge:yi1, xi0:xi1] *= 1.18            # light on the panel's lower lip
    img[yi0:yi1, xi0:xi0 + edge] *= 0.80
    img[yi0:yi1, xi1 - edge:xi1] *= 1.10
    for a, b in rails:
        r0, r1 = int(a * h), int(b * h)
        img[r0:r0 + 2] *= 1.15
        img[r1 - 2:r1] *= 0.75
    img[:, :2] *= 0.82
    return _u8(np.clip(img, 0, 1))


def cane(h, w, seed):
    """Woven rattan cane (an octagonal open weave) on a darker ground."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    p = 10.0
    a = np.abs(np.sin(xx * math.pi / p))
    b = np.abs(np.sin(yy * math.pi / p))
    c = np.abs(np.sin((xx + yy) * math.pi / (p * 1.414)))
    d = np.abs(np.sin((xx - yy) * math.pi / (p * 1.414)))
    strands = np.maximum.reduce([a ** 6, b ** 6, 0.85 * c ** 8, 0.85 * d ** 8])
    base = np.array([0.30, 0.20, 0.10])
    strand = np.array([0.86, 0.72, 0.48])
    t = np.clip(strands, 0, 1)
    img = base[None, None, :] * (1 - t[..., None]) + strand[None, None, :] * t[..., None]
    img *= (0.92 + 0.1 * noise(h, w, (8, 8), seed, 2))[..., None]
    return _u8(img)


def linen(h, w, colour, seed):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    weave = 0.5 + 0.25 * np.sin(xx * 1.3) + 0.25 * np.sin(yy * 1.3)
    slub = noise(h, w, (32, 4), seed, 2)
    return _u8(_col(colour) * (0.90 + 0.07 * weave[..., None] + 0.06 * slub[..., None]))


def brushed(h, w, base, seed):
    streak = noise(h, w, (h // 2, 4), seed, 2)
    return _u8(_col(base) * (0.93 + 0.09 * streak[..., None]))


def tiles(h, w, seed, rows=8, cols=4, tile=(0.92, 0.92, 0.90), grout=(0.70, 0.70, 0.68)):
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3)) * _col(grout)
    th, tw = h // rows, w // cols
    g = max(2, th // 10)
    for r in range(rows):
        off = (tw // 2) if r % 2 else 0
        for c in range(-1, cols + 1):
            x0 = c * tw + off
            xs, xe = max(0, x0 + g), min(w, x0 + tw - g)
            if xe <= xs:
                continue
            col = np.asarray(tile) * rng.uniform(0.97, 1.03)
            block = np.ones((th - 2 * g, xe - xs, 3)) * col[None, None, :]
            block[:2] *= 1.04
            block[-2:] *= 0.92
            img[r * th + g:r * th + th - g, xs:xe] = block
    return _u8(np.clip(img, 0, 1))


def terracotta(h, w, seed):
    n = noise(h, w, (6, 6), seed, 4)
    return _u8(_col((0.66, 0.36, 0.24)) * (0.85 + 0.25 * n[..., None]))


def ceramic(h, w, colour, seed):
    n = noise(h, w, (8, 3), seed, 3)
    return _u8(_col(colour) * (0.90 + 0.14 * n[..., None]))


# ================================================================================================ pictures and signs
def chalkboard(w=1024, h=768):
    """The lunch menu on a slate board in chalk."""
    rng = np.random.default_rng(91)
    base = np.ones((h, w, 3)) * _col((0.13, 0.15, 0.14))
    smear = noise(h, w, (5, 7), 92, 4)
    base *= (0.85 + 0.35 * smear[..., None])
    im = Image.fromarray(_u8(base))
    dr = ImageDraw.Draw(im)
    chalk = (226, 224, 214)
    dr.text((w // 2, 70), "Linden", font=font(86, serif=True), fill=chalk, anchor="mm")
    dr.text((w // 2, 150), "~ LUNCH ~", font=font(34), fill=(208, 196, 150), anchor="mm")
    items = [("Soupe du jour", "9"), ("Steak frites", "24"), ("Tagliatelle, wild mushrooms", "18"),
             ("Grilled trout, lemon butter", "22"), ("Salade nicoise", "16"), ("Tarte tatin", "8"),
             ("Espresso / Cafe creme", "3 / 4")]
    y = 230
    for name, price in items:
        dr.text((90, y), name, font=font(36, serif=True), fill=chalk, anchor="lm")
        dr.text((w - 90, y), price, font=font(36, serif=True), fill=chalk, anchor="rm")
        y += 66
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    arr = np.asarray(im).astype(np.float64) / 255.0
    arr *= (0.96 + 0.06 * rng.random((h, w)))[..., None]
    return _u8(arr)


def painting(kind, w=512, h=640, seed=0):
    """Framed art: an abstract colour-field, a soft landscape or a botanical study (painterly, low contrast)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    y, x = yy / h, xx / w
    n = noise(h, w, (6, 5), seed, 5)
    if kind == "field":
        c1, c2, c3 = np.array([0.78, 0.56, 0.32]), np.array([0.28, 0.36, 0.33]), np.array([0.88, 0.82, 0.70])
        t = np.clip((y - 0.45 + 0.08 * (n - 0.5)) * 6, 0, 1)
        img = c3[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]
        blob = ((x - 0.5) ** 2 / 0.09 + (y - 0.30) ** 2 / 0.03) < 1 + 0.4 * (n - 0.5)
        img[blob] = img[blob] * 0.3 + c2 * 0.7
    elif kind == "landscape":
        sky = np.array([0.80, 0.84, 0.86])
        hill1, hill2, field = np.array([0.42, 0.52, 0.40]), np.array([0.30, 0.40, 0.33]), np.array([0.76, 0.66, 0.42])
        img = np.ones((h, w, 3)) * sky
        img *= (0.92 + 0.12 * (1 - y))[..., None]
        h1 = 0.45 + 0.08 * np.sin(x * 5.0 + 1.0) + 0.04 * n
        h2 = 0.60 + 0.06 * np.sin(x * 7.0 + 2.5) + 0.04 * n
        img[y > h1] = hill1
        img[y > h2] = hill2
        img[y > 0.78 + 0.03 * np.sin(x * 9.0)] = field
        img *= (0.9 + 0.18 * n)[..., None]
    else:
        img = np.ones((h, w, 3)) * np.array([0.92, 0.89, 0.80])
        rng = np.random.default_rng(seed)
        for i in range(9):
            cx, cy = 0.5 + 0.22 * math.cos(i * 0.7), 0.25 + 0.06 * i
            a = rng.uniform(0, math.pi)
            u = (x - cx) * math.cos(a) + (y - cy) * math.sin(a)
            v = -(x - cx) * math.sin(a) + (y - cy) * math.cos(a)
            leaf = (u / 0.14) ** 2 + (v / 0.05) ** 2 < 1
            img[leaf] = np.array([0.30, 0.45, 0.30]) * rng.uniform(0.8, 1.15)
        stem = (np.abs(x - 0.5 - 0.05 * np.sin(y * 6)) < 0.006) & (y > 0.15) & (y < 0.92)
        img[stem] = np.array([0.35, 0.30, 0.20])
        img *= (0.94 + 0.08 * n)[..., None]
    im = Image.fromarray(_u8(np.clip(img, 0, 1))).filter(ImageFilter.GaussianBlur(1.6))
    return np.asarray(im)


def street(w=2048, h=1024, seed=5):
    """What the windows look out on, as a soft out-of-focus backdrop: the far pavement, a row of townhouses across
    the street (stone and painted fronts, windows, a cafe awning, shop fronts), street trees and the sky."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    y = yy / h
    img = np.ones((h, w, 3)) * np.array([0.74, 0.83, 0.92])
    img *= (0.88 + 0.16 * y)[..., None]
    x = 0
    facade_cols = [(0.84, 0.78, 0.66), (0.70, 0.60, 0.50), (0.88, 0.86, 0.80), (0.60, 0.66, 0.70),
                   (0.80, 0.70, 0.58), (0.74, 0.76, 0.72)]
    while x < w:
        bw = int(rng.integers(260, 420))
        top = int(rng.uniform(0.05, 0.18) * h)
        col = np.array(facade_cols[int(rng.integers(len(facade_cols)))])
        img[top:int(0.86 * h), x:x + bw] = col
        img[top:top + 10, x:x + bw] *= 0.8                       # cornice
        for r in range(4):
            wy = top + 40 + r * 120
            if wy + 70 > int(0.62 * h):
                break
            for cxi in range(3):
                wx = x + 40 + cxi * (bw - 80) // 3
                img[wy:wy + 70, wx:wx + 44] = np.array([0.28, 0.32, 0.38]) * rng.uniform(0.8, 1.2)
                img[wy + 68:wy + 74, wx - 4:wx + 48] = col * 1.08
        # ground floor: a shop front and maybe an awning
        img[int(0.64 * h):int(0.84 * h), x + 20:x + bw - 20] = np.array([0.22, 0.24, 0.27])
        if rng.random() < 0.6:
            aw = np.array([(0.55, 0.15, 0.12), (0.18, 0.34, 0.28), (0.80, 0.72, 0.50)][int(rng.integers(3))])
            img[int(0.60 * h):int(0.65 * h), x + 10:x + bw - 10] = aw
        img[top:int(0.86 * h), x:x + 3] *= 0.75
        x += bw
    img[int(0.84 * h):] = np.array([0.55, 0.55, 0.54])            # the far pavement
    img[int(0.90 * h):] = np.array([0.40, 0.40, 0.41])            # the street
    for tx in range(120, w, 420):                                 # street trees
        cx = tx + int(rng.integers(-40, 40))
        r = int(rng.integers(110, 150))
        d = ((xx - cx) / r) ** 2 + ((yy - 0.42 * h) / (r * 1.1)) ** 2
        crown = d < 1.0 + 0.3 * (noise(h, w, (8, 16), seed + tx, 3) - 0.5)
        img[crown] = np.array([0.30, 0.42, 0.24]) * (0.8 + 0.4 * noise(h, w, (20, 40), seed + tx + 1, 2))[crown][:, None]
        img[int(0.55 * h):int(0.88 * h), cx - 8:cx + 8] = np.array([0.30, 0.24, 0.18])
    im = Image.fromarray(_u8(np.clip(img, 0, 1))).filter(ImageFilter.GaussianBlur(5.0))
    return np.asarray(im)


def cone_label(w=512, h=256):
    """The wet-floor cone's printed band (twice round the cone): CAUTION / WET FLOOR and the slip pictogram."""
    im = Image.new("RGB", (w, h), (242, 196, 28))
    dr = ImageDraw.Draw(im)
    for k in range(2):
        cx = w // 4 + k * w // 2
        dr.text((cx, 40), "CAUTION", font=font(40, bold=True), fill=(20, 20, 20), anchor="mm")
        dr.text((cx, 215), "WET FLOOR", font=font(34, bold=True), fill=(20, 20, 20), anchor="mm")
        # the pictogram: a triangle with a slipping figure
        tri = [(cx - 60, 175), (cx + 60, 175), (cx, 75)]
        dr.polygon(tri, outline=(20, 20, 20), width=6)
        dr.ellipse((cx - 8, 105, cx + 6, 119), fill=(20, 20, 20))
        dr.line((cx - 2, 120, cx - 12, 148), fill=(20, 20, 20), width=6)
        dr.line((cx - 12, 148, cx - 30, 160), fill=(20, 20, 20), width=5)
        dr.line((cx - 12, 148, cx + 12, 162), fill=(20, 20, 20), width=5)
        dr.line((cx - 4, 128, cx + 18, 122), fill=(20, 20, 20), width=5)
        dr.line((cx - 30, 168, cx + 32, 168), fill=(20, 20, 20), width=3)
    return np.asarray(im)


def sign(text, w=512, h=128, bg=(0.10, 0.12, 0.11), fg=(0.86, 0.76, 0.48), serif=True, size=56):
    im = Image.new("RGB", (w, h), tuple(int(c * 255) for c in bg))
    dr = ImageDraw.Draw(im)
    dr.text((w // 2, h // 2), text, font=font(size, serif=serif), fill=tuple(int(c * 255) for c in fg), anchor="mm")
    return np.asarray(im.filter(ImageFilter.GaussianBlur(0.6)))


def kitchen_back(w=512, h=512):
    """What shows through the pass: white subway tiles, a steel shelf line."""
    img = tiles(h, w, 61).astype(np.float64) / 255.0
    img[int(0.45 * h):int(0.48 * h)] = np.array([0.55, 0.56, 0.58])
    return _u8(img * 0.92)


# ================================================================================================ registry
_T = {
    "floor": (chevron, ()),
    "plaster": (plaster, (512, 512, (0.90, 0.85, 0.76), 11)),
    "ceiling": (plaster, (256, 256, (0.93, 0.91, 0.87), 12)),
    "wainscot": (wainscot, (640, 512, (0.16, 0.27, 0.22), 13)),
    "walnut": (wood, (512, 512, (0.46, 0.30, 0.19), (0.27, 0.16, 0.09), 14)),
    "oak": (wood, (512, 512, (0.78, 0.62, 0.42), (0.58, 0.41, 0.25), 15)),
    "beam": (wood, (256, 512, (0.36, 0.25, 0.16), (0.20, 0.13, 0.08), 16, 18.0)),
    "bentwood": (wood, (128, 256, (0.30, 0.19, 0.11), (0.17, 0.10, 0.06), 17, 12.0)),
    "cane": (cane, (256, 256, 18)),
    "linen_white": (linen, (256, 256, (0.93, 0.92, 0.88), 19)),
    "linen_sage": (linen, (256, 256, (0.52, 0.60, 0.50), 20)),
    "linen_curtain": (linen, (256, 256, (0.90, 0.87, 0.80), 21)),
    "kraft": (plaster, (64, 64, (0.74, 0.60, 0.40), 22)),
    "stainless": (brushed, (256, 256, (0.72, 0.73, 0.74), 23)),
    "terracotta": (terracotta, (256, 256, 24)),
    "pot_ceramic": (ceramic, (256, 256, (0.20, 0.30, 0.30), 25)),
    "chalkboard": (chalkboard, ()),
    "art_field": (painting, ("field", 512, 640, 31)),
    "art_landscape": (painting, ("landscape", 640, 512, 32)),
    "art_botanical": (painting, ("botanical", 512, 640, 33)),
    "street": (street, ()),
    "cone_label": (cone_label, ()),
    "sign_dish": (sign, ("DISH RETURN", 512, 96, (0.12, 0.13, 0.13), (0.92, 0.92, 0.90), False, 50)),
    "sign_linden": (sign, ("Linden", 768, 192, (0.16, 0.27, 0.22), (0.86, 0.74, 0.45), True, 120)),
    "kitchen": (kitchen_back, ()),
    "mop_pad": (linen, (128, 128, (0.20, 0.42, 0.78), 26)),
}


@functools.lru_cache(maxsize=None)
def texture(name: str) -> np.ndarray:
    fn, args = _T[name]
    img = fn(*args)
    return np.ascontiguousarray(img[..., :3])


TEXTURE_NAMES = tuple(_T)
