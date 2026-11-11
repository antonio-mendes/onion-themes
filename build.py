#!/usr/bin/env python3
"""Generates the Landform and Keep Onion OS themes (skins, charging animation, sounds, config).

Usage: python3 build.py OUT_DIR [AUTHOR]
"""
import json, math, os, shutil, subprocess, sys, wave, zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
AUTHOR = sys.argv[2] if len(sys.argv) > 2 else "antonio-mendes"
FONTS = {
    "Caladea-Bold.ttf": "/usr/share/fonts/truetype/crosextra/Caladea-Bold.ttf",
    "Caladea-Italic.ttf": "/usr/share/fonts/truetype/crosextra/Caladea-Italic.ttf",
    "Caladea-Regular.ttf": "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf",
    "JetBrainsMono-SemiBold.ttf": "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-SemiBold.ttf",
    "JetBrainsMono-Regular.ttf": "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
    "DejaVuSans-Bold.ttf": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
}
W, H = 640, 480
FRAMES = 24
SS = 4

# ---------------------------------------------------------------- helpers
def rgb(h, a=255):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)

def mix(a, b, t):
    A, B = rgb(a), rgb(b)
    return "#%02x%02x%02x" % tuple(round(A[i] + (B[i] - A[i]) * t) for i in range(3))

def ramp(stops, t):
    t = min(0.9999, max(0.0, t)) * (len(stops) - 1)
    i = int(t)
    return mix(stops[i], stops[i + 1], t - i)

def font(name, size):
    return ImageFont.truetype(FONTS[name], size)

def new(w, h, fill=None):
    return Image.new("RGBA", (w, h), rgb(fill) if isinstance(fill, str) else (fill or (0, 0, 0, 0)))

def down(img, w, h):
    return img.convert("RGBa").resize((w, h), Image.LANCZOS).convert("RGBA")

def hashn(x, y, s):
    h = (x.astype(np.int64) * 374761393 + y.astype(np.int64) * 668265263 + s * 2246822519) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return h / 4294967296.0

def vnoise(x, y, s):
    xi, yi = np.floor(x), np.floor(y)
    fx, fy = x - xi, y - yi
    u, v = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = hashn(xi, yi, s), hashn(xi + 1, yi, s)
    c, d = hashn(xi, yi + 1, s), hashn(xi + 1, yi + 1, s)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v

def fbm(x, y, s, o=4):
    x, y = np.asarray(x, float), np.asarray(y, float)
    v, a, f, n = 0, 0.5, 1, 0
    for i in range(o):
        v = v + a * vnoise(x * f, y * f, s + i)
        n += a; a *= 0.5; f *= 2
    return v / n

BAYER = (np.array([0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]).reshape(4, 4) + 0.5) / 16

def stipple(density, cell, color, into=None, at=(0, 0)):
    """Ordered-dither a density map (rows x cols, values <0 = empty) into ink pixels."""
    r, c = density.shape
    th = np.tile(BAYER, (r // 4 + 1, c // 4 + 1))[:r, :c]
    mask = np.repeat(np.repeat(density > th, cell, 0), cell, 1)
    layer = np.zeros((r * cell, c * cell, 4), np.uint8)
    layer[mask] = rgb(color) if isinstance(color, str) else color
    img = Image.fromarray(layer, "RGBA")
    if into is None:
        return img
    into.alpha_composite(img, at)
    return into

def text(img, xy, s, fnt, fill, anchor="mm"):
    ImageDraw.Draw(img).text(xy, s, font=fnt, fill=rgb(fill) if isinstance(fill, str) else fill, anchor=anchor)

def frame(img, box, color, w=2, fill=None):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    if fill is not None:
        d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=rgb(fill) if isinstance(fill, str) else fill)
    if color:
        for i in range(w):
            d.rectangle((x0 + i, y0 + i, x1 - 1 - i, y1 - 1 - i), outline=rgb(color))

# ---------------------------------------------------------------- small glyph icons
class Pen:
    """Draws in unit coordinates (0..1) on a supersampled canvas."""
    def __init__(self, w, h, color):
        self.w, self.h, self.c = w, h, rgb(color)
        self.img = Image.new("RGBA", (w * SS, h * SS), self.c[:3] + (0,))
        self.d = ImageDraw.Draw(self.img)
        self.s = min(w, h) * SS
        self.ox, self.oy = (w * SS - self.s) / 2, (h * SS - self.s) / 2
    def p(self, x, y): return (self.ox + x * self.s, self.oy + y * self.s)
    def line(self, pts, w=.09, closed=False):
        P = [self.p(*q) for q in pts]
        if closed: P.append(P[0])
        self.d.line(P, fill=self.c, width=max(1, round(w * self.s)), joint="curve")
        r = w * self.s / 2
        for (x, y) in P: self.d.ellipse((x - r, y - r, x + r, y + r), fill=self.c)
    def circle(self, x, y, r, w=None, erase=False):
        cx, cy = self.p(x, y); r *= self.s
        col = self.c[:3] + (0,) if erase else self.c
        if w is None: self.d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
        else: self.d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=col, width=max(1, round(w * self.s)))
    def arc(self, x, y, r, a0, a1, w=.09):
        cx, cy = self.p(x, y); r *= self.s
        self.d.arc((cx - r, cy - r, cx + r, cy + r), a0, a1, fill=self.c, width=max(1, round(w * self.s)))
    def poly(self, pts, erase=False):
        self.d.polygon([self.p(*q) for q in pts], fill=self.c[:3] + (0,) if erase else self.c)
    def rect(self, x0, y0, x1, y1, w=None, rad=0, erase=False):
        a, b = self.p(x0, y0), self.p(x1, y1)
        col = self.c[:3] + (0,) if erase else self.c
        if w is None: self.d.rounded_rectangle((*a, *b), radius=rad * self.s, fill=col)
        else: self.d.rounded_rectangle((*a, *b), radius=rad * self.s, outline=col, width=max(1, round(w * self.s)))
    def out(self): return down(self.img, self.w, self.h)

def wifi(p, n=3, y=.78):
    p.circle(.5, y, .07)
    for i in range(n):
        p.arc(.5, y, .2 + i * .17, 225, 315, .085)

def g_shutdown(p): p.arc(.5, .54, .34, -60, 240, .1); p.line([(.5, .12), (.5, .5)], .1)
def g_tf(p): p.poly([(.2, .05), (.62, .05), (.8, .25), (.8, .95), (.2, .95)]); [p.rect(.3 + i * .14, .12, .37 + i * .14, .3, erase=True) for i in range(3)]
def g_brightness(p):
    p.circle(.5, .5, .17)
    for k in range(8):
        a = k * math.pi / 4
        p.line([(.5 + math.cos(a) * .3, .5 + math.sin(a) * .3), (.5 + math.cos(a) * .41, .5 + math.sin(a) * .41)], .075)
def g_info(p): p.circle(.5, .5, .4, .085); p.line([(.5, .46), (.5, .7)], .1); p.circle(.5, .3, .06)
def g_reset(p): p.arc(.5, .52, .32, -40, 250, .095); p.poly([(.5, .02), (.76, .2), (.5, .38)])
def g_folder(p): p.rect(.1, .28, .9, .82, rad=.05); p.poly([(.1, .32), (.1, .2), (.42, .2), (.5, .32)])
def g_hotspot(p): p.circle(.5, .5, .1); p.arc(.5, .5, .27, 130, 230, .08); p.arc(.5, .5, .27, -50, 50, .08); p.arc(.5, .5, .43, 135, 225, .08); p.arc(.5, .5, .43, -45, 45, .08)
def g_keys(p): p.rect(.38, .12, .62, .88, rad=.04); p.rect(.12, .38, .88, .62, rad=.04); p.circle(.5, .5, .07, erase=True)
def g_lang(p): p.circle(.5, .5, .4, .075); p.line([(.1, .5), (.9, .5)], .06); p.d.ellipse((*p.p(.31, .1), *p.p(.69, .9)), outline=p.c, width=round(.06 * p.s))
def g_left(p): p.line([(.64, .16), (.3, .5), (.64, .84)], .14)
def g_right(p): p.line([(.36, .16), (.7, .5), (.36, .84)], .14)
def g_wifi(p): wifi(p)
def g_wifi_ok(p): wifi(p, 3, .7); p.line([(.62, .84), (.72, .93), (.92, .72)], .08)
def g_wifi_lock(p): wifi(p, 3, .7); p.rect(.66, .74, .94, .96, rad=.03); p.arc(.8, .75, .09, 180, 360, .05)
def g_theme(p):
    p.poly([(.06, .86), (.36, .3), (.52, .58), (.64, .42), (.94, .86)]); p.circle(.76, .22, .1)
def g_sound(p): p.poly([(.1, .38), (.28, .38), (.5, .18), (.5, .82), (.28, .62), (.1, .62)]); p.arc(.5, .5, .2, -50, 50, .075); p.arc(.5, .5, .36, -50, 50, .075)
def g_star(p):
    p.poly([(.5 + math.cos(-math.pi / 2 + k * math.pi / 5) * (.2 if k % 2 else .46), .52 + math.sin(-math.pi / 2 + k * math.pi / 5) * (.2 if k % 2 else .46)) for k in range(10)])
def g_sig(n):
    def f(p):
        for i in range(4):
            x = .14 + i * .2
            if i < n: p.rect(x, .78 - i * .2, x + .13, .9)
            else: p.rect(x, .84, x + .13, .9)
    return f

GLYPHS = {
    "icon-Shutdown": (47, 46, g_shutdown), "icon-TF": (19, 24, g_tf), "icon-brightness-48": (48, 48, g_brightness),
    "icon-device-info-48": (48, 48, g_info), "icon-factory-reset-48": (48, 49, g_reset), "icon-folder": (60, 60, g_folder),
    "icon-hotspot-wifi": (35, 35, g_hotspot), "icon-key-setting-48": (48, 48, g_keys), "icon-language-48": (48, 48, g_lang),
    "icon-left-arrow-24": (24, 24, g_left), "icon-right-arrow-24": (24, 24, g_right), "icon-setting-wifi": (46, 46, g_wifi),
    "icon-theme": (46, 46, g_theme), "color": (44, 44, g_theme), "icon-wifi-connected": (48, 48, g_wifi_ok),
    "icon-wifi-locked": (48, 48, g_wifi_lock), "sound-icon": (48, 48, g_sound),
    "icon-wifi-signal-01": (46, 46, g_sig(1)), "icon-wifi-signal-02": (46, 46, g_sig(2)),
    "icon-wifi-signal-03": (46, 46, g_sig(3)), "icon-wifi-signal-04": (46, 46, g_sig(4)),
}

def glyph(fn, w, h, color):
    p = Pen(w, h, color); fn(p); return p.out()

# ---------------------------------------------------------------- Landform art
def lf_field(t, c, full=False):
    """Dither wave field at cell size 2. t in [0,1) loops seamlessly."""
    rows, cols = H // 2, W // 2
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    T = 2 * math.pi * t
    w = (np.sin(x * .045 + c["ph"] + T) + np.sin(y * .08 - 1 - T) + np.sin((x + y) * .028 + .7 + T)
         + np.sin(np.hypot(x - c["cx"], y - c["cy"]) * .09 - 2.4 - 2 * T))
    s = np.clip(w / 2.4 + .5, 0, 1); s = s * s * (3 - 2 * s)
    if full:
        return s * .85
    yy = y * 2
    edge = np.maximum(np.where(yy < 150, np.clip(1 - (yy - 56) / 94, 0, 1) ** 2, 0),
                      np.where(yy > 330, np.clip((yy - 330) / 92, 0, 1) ** 2, 0))
    edge = np.maximum(edge * c["fmax"], c.get("floor", 0))
    return s * edge

def lf_icon_density(i, n=60):
    v, u = (np.mgrid[0:n, 0:n] + .5) / n
    out = np.full((n, n), -1.0)
    if i == 0:  # strata (recents)
        w = v + np.sin(u * 7) * .04
        b = np.clip(np.floor(w * 5), 0, 4).astype(int)
        d = np.array([.12, .32, .55, .78, 1])[b] * ((w * 5 - np.floor(w * 5)) >= .12)
        out = np.where((v >= .08) & (v <= .94), d, -1)
    elif i == 1:  # peak (favorites)
        pk = .5 + np.sin(v * 9) * .02; hw = (v - .06) * .56
        ins = (v >= .06) & (v <= .94) & (np.abs(u - pk) <= hw)
        d = np.where(v < .3, np.where(u < pk, .05, .3), np.where(u < pk, .22 + (v - .3) * .3, .72 + (v - .3) * .4))
        out = np.where(ins, d, -1)
    elif i == 2:  # crater (games)
        dx, dy = u - .5, (v - .5) * 1.5; r = np.hypot(dx, dy)
        d = np.where(r < .17, .95, np.where(r < .25, .08, .25 + (dx * .7 + dy * .5 + .4) * .7))
        out = np.where(r <= .48, d, -1)
    elif i == 3:  # plots (apps)
        cu, cv = (u >= .5).astype(int), (v >= .5).astype(int)
        lu, lv = u * 2 - cu, v * 2 - cv
        d = np.array([[.2, .55], [.8, 1]])[cv, cu]
        out = np.where((lu >= .1) & (lu <= .9) & (lv >= .1) & (lv <= .9), d, -1)
    elif i == 4:  # mesa (expert)
        hw = .27 + (v - .22) * .16
        body = (v >= .22) & (v <= .88) & (np.abs(u - .5) <= hw)
        cren = (v >= .12) & (v < .22) & (np.abs(u - .5) <= .27) & ((np.floor((u - .23) / .09) % 2) == 0)
        ground = (v > .88) & (v <= .94) & (np.abs(u - .5) <= .47)
        d = np.where(v < .32, .14, np.where(np.sin(u * 46) > .2, .85, .5) * np.where(u < .5, .62, 1))
        out = np.where(ground, 1.0, np.where(cren, .14, np.where(body, d, -1)))
    else:  # core samples (settings)
        k = np.clip(np.floor(u * 3), 0, 2).astype(int); lu = u * 3 - k
        knob = np.array([.3, .62, .44])[k]
        ins = (lu >= .16) & (lu <= .84) & (v >= .07) & (v <= .93)
        d = np.where(np.abs(v - knob) < .055, 1.0, np.where(v < knob, .1, .62))
        edge = (lu < .21) | (lu > .79) | (v < .095) | (v > .905)
        out = np.where(ins, np.where(edge, 1.0, d), -1)
    return out

# ---------------------------------------------------------------- Keep art
def col(d, ox, oy, a, x, y, z0, z1, c):
    sx, sy = ox + (x - y) * a, oy + (x + y) * a / 2
    t, b = sy - z1 * a, sy - z0 * a
    d.polygon([(sx, t), (sx + a, t + a / 2), (sx, t + a), (sx - a, t + a / 2)], fill=rgb(c[0]))
    if z1 > z0:
        d.polygon([(sx - a, t + a / 2), (sx, t + a), (sx, b + a), (sx - a, b + a / 2)], fill=rgb(c[1]))
        d.polygon([(sx + a, t + a / 2), (sx, t + a), (sx, b + a), (sx + a, b + a / 2)], fill=rgb(c[2]))

VOX = [
    [(0, 0, 0, 1), (1, 0, 0, 2), (2, 0, 0, 3), (0, 1, 0, 1), (1, 1, 0, 2), (2, 1, 0, 3)],  # stairs (recents)
    [(x, y, 0, 1) for y in range(3) for x in range(3)] + [(.5, .5, 1, 2), (1.5, .5, 1, 2), (.5, 1.5, 1, 2), (1.5, 1.5, 1, 2), (1, 1, 2, 3.2, 1)],  # pyramid (favorites)
    [(x, y, 0, .7) for y in range(2) for x in range(4)] + [(.5, .5, .7, 1.2, 1), (2.6, .2, .7, 1.2, 1), (2.9, .9, .7, 1.2, 1)],  # pad (games)
    [(0, 0, 0, 1), (1.6, 0, 0, 1), (0, 1.6, 0, 1), (1.6, 1.6, 0, 1)],  # cubes (apps)
    [(x, y, 0, .6) for y in range(3) for x in range(3)] + [(1, 1, .6, 3.1), (1, 1, 3.1, 3.7, 1)],  # tower (expert)
    [(0, 0, 0, 1.4), (0, 0, 1.4, 1.9, 1), (1.5, 0, 0, 2.6), (1.5, 0, 2.6, 3.1, 1), (3, 0, 0, .8), (3, 0, .8, 1.3, 1)],  # sliders (settings)
]
VOX_FIT = [(17, -2, 6), (15, 0, 4), (15, -9, 14), (18, 0, 4), (15, 0, 12), (17, -17, 16)]  # scale, dx, dy

def keep_icon(i, c, selected, w=132, h=150):
    img = new(w * SS, h * SS); d = ImageDraw.Draw(img)
    a, dx, dy = VOX_FIT[i]
    for v in sorted(VOX[i], key=lambda v: (v[2], v[0] + v[1])):
        pal = (c.get("accSel", c["acc"]) if selected else c["acc"]) if len(v) > 4 and v[4] else (c["blockSel"] if selected else c["block"])
        col(d, (w / 2 + dx) * SS, (h / 2 - 18 + dy) * SS, a * SS, v[0] - 1, v[1] - 1, v[2], v[3], pal)
    return down(img, w, h)

def keep_heights(c, t=None):
    N = 28
    y, x = np.mgrid[0:N, 0:N].astype(float)
    f = np.clip((fbm(x * .095, y * .095, c["seed"], 3) - .36) / .42, 0, None)
    h = f * f * 6 * 1.6
    if t is not None:
        h = h + .9 * np.sin(2 * math.pi * t - (x + y) * .32) * (h > .4)
    return np.clip(np.round(h * 2) / 2, 0, 6)

def keep_terrain(c, t=None):
    S = 2
    img = new(W * S, H * S, c["bg"]); d = ImageDraw.Draw(img)
    hs = keep_heights(c, t); N = hs.shape[0]
    for s in range(2 * N - 1):
        for x in range(max(0, s - N + 1), min(N - 1, s) + 1):
            y = s - x; h = float(hs[y, x]); tt = h / 6
            if h < .5:
                pal = (c["water"], mix(c["water"], "#000000", .25), mix(c["water"], "#000000", .45)); z = .25
            else:
                side = ramp(c["side"], tt)
                pal = (ramp(c["terrain"], tt), mix(side, "#000000", .28), mix(side, "#000000", .5)); z = h
            col(d, 320 * S, 62 * S, 13 * S, x, y, 0, z, pal)
    img = down(img, W, H)
    veil = new(W, H, rgb(c["bg"], round(255 * c["veil"])))
    img.alpha_composite(veil)
    return img

# ---------------------------------------------------------------- per-theme pieces
def background(c, t=0.0, charging=False):
    if c["kind"] == "landform":
        img = new(W, H, c["bg"])
        stipple(lf_field(t, c, full=charging), 2, c["field"], img)
        return img
    return keep_terrain(c, t if charging else None)

TILE = (12, 30, 144, 262)  # tile box inside the 156x328 main-menu icon
def main_icon(i, c, selected):
    img = new(156, 328)
    x0, y0, x1, y1 = TILE
    if c["kind"] == "landform":
        if selected: frame(img, TILE, None, fill=c["sel"])
        elif c.get("frames"): frame(img, TILE, c["dim"], 2, fill=c["bg"])
        else: frame(img, TILE, None, fill=rgb(c["bg"], 232))
        stipple(lf_icon_density(i), 2, c["selInk"] if selected else c["iconInk"], img, (78 - 60, 52))
    else:
        if selected: frame(img, TILE, None, fill=c["sel"])
        elif c.get("tiles"): frame(img, TILE, c["dim"], 1, fill=rgb(c["bg"], 226))
        else:
            sh = new(156 * SS, 328 * SS); d = ImageDraw.Draw(sh)
            d.ellipse((22 * SS, 128 * SS, 134 * SS, 170 * SS), fill=rgb(c["bg"], 190))
            d.rounded_rectangle((14 * SS, 200 * SS, 142 * SS, 242 * SS), radius=6 * SS, fill=rgb(c["bg"], 215))
            img.alpha_composite(down(sh, 156, 328))
        img.alpha_composite(keep_icon(i, c, selected), (12, 44))
    return img

def bar(c, h, rule_at_bottom):
    img = new(W, h, c["bg"])
    ImageDraw.Draw(img).rectangle((0, h - 2, W, h) if rule_at_bottom else (0, 0, W, 1), fill=rgb(c["dim"]))
    return img

def highlight(c, w, h):
    img = new(w, h, c["hl"])
    if c.get("hlStipple"):
        stipple(np.full((h // 2, w // 2), c["hlStipple"]), 2, c["field"], img)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, 1), fill=rgb(c["dim"])); d.rectangle((0, h - 2, w, h), fill=rgb(c["dim"]))
    d.rectangle((0, 0, 7, h), fill=rgb(c["accent"]))
    return img

def panel(c, w, h, box=None, hl=False, fw=2):
    img = new(w, h)
    box = box or (0, 0, w, h)
    frame(img, box, c["accent"] if hl else c["dim"], fw, fill=c["hl"] if hl else c["panel"])
    return img

def battery_small(c, level, bolt=False):
    p = Pen(88, 48, c["ink"]); s = SS
    d = p.d
    d.rounded_rectangle((2 * s, 10 * s, 17 * s, 41 * s), radius=3 * s, outline=p.c, width=2 * s)
    d.rectangle((6 * s, 7 * s, 13 * s, 10 * s), fill=p.c)
    if level > 0:
        top = 37 - round(23 * level)
        d.rectangle((6 * s, top * s, 13 * s, 37 * s), fill=rgb(c["low"] if level <= .2 else c["accent"]))
    if bolt:
        d.polygon([(11 * s, 15 * s), (6 * s, 27 * s), (9.5 * s, 27 * s), (8 * s, 36 * s), (13 * s, 24 * s), (9.5 * s, 24 * s)], fill=p.c)
    return down(p.img, 88, 48)

def battery_big(c, w, h, bars, total=4, ink=None, fill=None):
    """Horizontal battery, used for the charging screens."""
    img = new(w * SS, h * SS); d = ImageDraw.Draw(img); s = SS
    ink = rgb(ink or c["ink"]); lw = max(2, h // 16)
    bw = w - h // 5
    d.rounded_rectangle((0, 0, bw * s, h * s - 1), radius=h * s // 7, outline=ink, width=lw * s)
    d.rounded_rectangle((bw * s, h * s * .3, w * s - 1, h * s * .7), radius=2 * s, fill=ink)
    pad = lw * 2; cw = (bw - pad * 2 - (total - 1) * lw) / total
    for k in range(bars):
        x = pad + k * (cw + lw)
        d.rectangle((x * s, pad * s, (x + cw) * s, (h - pad) * s - 1), fill=rgb(fill or c["accent"]))
    return down(img, w, h)

def pill(c, label, w, h, fnt):
    p = Pen(w, h, c["ink"])
    p.d.rounded_rectangle((0, 0, w * SS - 1, h * SS - 1), radius=7 * SS, fill=p.c)
    p.d.ellipse((9 * SS, (h / 2 - 5) * SS, 19 * SS, (h / 2 + 5) * SS), fill=rgb(c["accentOnInk"]))
    img = p.out()
    text(img, (26, h // 2), label, fnt, c["bg"], "lm")
    return img

def button(c, letter, filled):
    p = Pen(32, 32, c["accent"] if filled else c["ink"])
    if filled: p.circle(.5, .5, .47)
    else: p.circle(.5, .5, .47, .08)
    img = p.out()
    text(img, (16, 16), letter, font(c["fBold"], 18), c["onAccent"] if filled else c["ink"])
    return img

def plate_screen(c, title, sub=None, art=None):
    img = background(c)
    box = (140, 150, 500, 330)
    frame(img, box, c["dim"], 2, fill=c["bg"])
    if art is not None:
        img.alpha_composite(art, (320 - art.width // 2, 176))
        text(img, (320, 268), title, font(c["fTitle"], 28), c["ink"])
    else:
        text(img, (320, 226 if sub else 240), title, font(c["fTitle"], 40), c["ink"])
    if sub: text(img, (320, 300 if art is not None else 276), sub, font(c["fGrid"], 18), c["dim2"])
    return img

LABELS = ["Recents", "Favorites", "Games", "Apps", "Expert", "Settings"]
ICON_FILES = ["recent", "favorite", "game", "app", "retroarch", "setting"]

# ---------------------------------------------------------------- sound
def write_wav(path, sig, sr):
    sig = np.clip(sig, -1, 1)
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((sig * 32767).astype("<i2").tobytes())

def make_change(path, c):
    sr = 44100; n = int(sr * .075); t = np.arange(n) / sr
    f0, f1 = c["tick"]
    env = np.exp(-t * 70)
    sig = np.sin(2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * .075))) * env
    rng = np.random.default_rng(1)
    sig += rng.standard_normal(n) * np.exp(-t * 400) * .25
    sig[:40] *= np.linspace(0, 1, 40)
    write_wav(path, sig * .5, sr)

def make_bgm(path, c):
    """Ambient loop: a walk across the theme's own terrain, mapped onto a scale."""
    sr = 48000; m = c["music"]; steps = m["steps"]; dur = steps * m["step"]
    N = int(sr * dur); buf = np.zeros(N)
    xs = np.arange(steps)
    ang = 2 * math.pi * xs / steps  # walk a circle through the noise so the tune loops
    hgt = fbm(3 + np.cos(ang) * 1.6, 3 + np.sin(ang) * 1.6, m["seed"], 3)
    hgt = (hgt - hgt.min()) / (hgt.max() - hgt.min() + 1e-9)
    scale = m["scale"]; notes = [scale[k % len(scale)] + 12 * (k // len(scale)) for k in range(len(scale) * 2 + 1)]
    def add(start, sig):
        idx = (int(start * sr) + np.arange(len(sig))) % N
        np.add.at(buf, idx, sig)
    for i in range(steps):
        if m["kind"] == "pluck" and hgt[i] < .22 and i % 4: continue
        midi = m["root"] + notes[int(round(hgt[i] * (len(notes) - 1)))]
        f = 440 * 2 ** ((midi - 69) / 12)
        if m["kind"] == "pad":
            L = m["step"] * 3.2; t = np.arange(int(L * sr)) / sr
            env = np.sin(math.pi * np.clip(t / L, 0, 1)) ** 2
            sig = (np.sin(2 * math.pi * f * t) + .25 * np.sin(4 * math.pi * f * t + .3) + .5 * np.sin(2 * math.pi * f * 1.003 * t)) * env * .22
        else:
            L = 1.8; t = np.arange(int(L * sr)) / sr
            env = np.exp(-t * 3.2) * np.clip(t / .004, 0, 1)
            sig = (np.sin(2 * math.pi * f * t) + .4 * np.sin(2 * math.pi * 2 * f * t) * np.exp(-t * 9) + .15 * np.sin(2 * math.pi * 3.01 * f * t) * np.exp(-t * 14)) * env * .3
        add(i * m["step"], sig)
    t = np.arange(N) / sr
    fr = 440 * 2 ** ((m["root"] - 12 - 69) / 12)
    k = round(fr * dur) / dur  # whole number of cycles so the drone loops cleanly
    buf += (np.sin(2 * math.pi * k * t) + .5 * np.sin(2 * math.pi * round(fr * 1.5 * dur) / dur * t)) * (.07 + .03 * np.sin(2 * math.pi * 2 * t / dur))
    buf = buf / np.max(np.abs(buf)) * .45
    tmp = path[:-4] + ".tmp.wav"
    write_wav(tmp, buf, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ar", "48000", "-ac", "1", "-b:a", "96k", path], check=True)
    os.remove(tmp)

# ---------------------------------------------------------------- themes
THEMES = [
    dict(name="Landform Paper", word="landform", kind="landform", desc="Dithered terrain on paper",
         bg="#f3efe3", ink="#2b2924", dim="#56704c", dim2="#56704c", field="#8aa07e", iconInk="#33442d", sel="#33442d", selInk="#f3efe3",
         accent="#33442d", onAccent="#f3efe3", accentOnInk="#a9c19c", low="#b8482c", hl="#f3efe3", hlStipple=.3, panel="#f3efe3", frames=True,
         ph=1.5, cx=220, cy=60, fmax=.62,
         fTitle="Caladea-Bold.ttf", fGrid="Caladea-Italic.ttf", fList="Caladea-Regular.ttf", fBold="Caladea-Bold.ttf", sizes=(27, 25, 25, 24),
         tick=(1500, 900), music=dict(kind="pad", root=57, scale=[0, 2, 4, 7, 9], steps=32, step=2.0, seed=11)),
    dict(name="Landform Night", word="landform", kind="landform", desc="Dithered terrain after dark",
         bg="#161a14", ink="#efe9d8", dim="#4f6446", dim2="#8aa07e", field="#4f6446", iconInk="#efe9d8", sel="#efe9d8", selInk="#161a14",
         accent="#efe9d8", onAccent="#161a14", accentOnInk="#4f6446", low="#e0704a", hl="#33412d", panel="#1d221a", frames=False,
         ph=4.2, cx=90, cy=150, fmax=.8, floor=.16,
         fTitle="JetBrainsMono-SemiBold.ttf", fGrid="JetBrainsMono-SemiBold.ttf", fList="JetBrainsMono-Regular.ttf", fBold="JetBrainsMono-SemiBold.ttf", sizes=(24, 19, 22, 20),
         tick=(1100, 700), music=dict(kind="pad", root=52, scale=[0, 3, 5, 7, 10], steps=32, step=2.25, seed=23)),
    dict(name="Keep Sandstone", word="keep", kind="keep", desc="Isometric block terrain in sand and teal",
         bg="#17130f", ink="#f2dcb0", dim="#8a5638", dim2="#c98f5a", sel="#f2dcb0", selInk="#17130f",
         accent="#3fa68f", onAccent="#17130f", accentOnInk="#3fa68f", low="#e0704a", hl="#4a3020", panel="#1f1913", tiles=True, seed=77, veil=.5,
         terrain=["#3a2e22", "#57432f", "#7a5b3c", "#a67c4f"], side=["#3a2e22", "#7a5b3c"], water="#24404a",
         block=("#f2dcb0", "#c98f5a", "#8a5638"), blockSel=("#3a2413", "#17130f", "#5b3a22"), acc=("#9fe3d0", "#3fa68f", "#256b5e"),
         fTitle="DejaVuSans-Bold.ttf", fGrid="DejaVuSans-Bold.ttf", fList="DejaVuSans.ttf", fBold="DejaVuSans-Bold.ttf", sizes=(24, 19, 22, 20),
         tick=(520, 300), music=dict(kind="pluck", root=50, scale=[0, 2, 3, 5, 7, 9, 10], steps=64, step=.75, seed=37)),
    dict(name="Keep Ember", word="keep", kind="keep", desc="Isometric block terrain coloured by elevation",
         bg="#0b0812", ink="#ffe9a8", dim="#5a2c7e", dim2="#a3308a", sel="#ffe9a8", selInk="#0b0812",
         accent="#ff7a45", onAccent="#0b0812", accentOnInk="#ff7a45", low="#e0456a", hl="#4b2569", panel="#150f22", tiles=False, seed=9, veil=.4,
         terrain=["#2b2350", "#5a2c7e", "#a3308a", "#e0456a", "#ff7a45", "#ffc46b"], side=["#1c1836", "#3a1f5c", "#6e2468", "#a2334f", "#b85a2c"], water="#141a3a",
         block=("#fff3d0", "#e9b98a", "#a8736a"), blockSel=("#2b2350", "#0b0812", "#3a1f5c"), acc=("#7df0e0", "#2fb5b0", "#1c7480"), accSel=("#ff7a45", "#d9502c", "#a2334f"),
         fTitle="JetBrainsMono-SemiBold.ttf", fGrid="JetBrainsMono-SemiBold.ttf", fList="JetBrainsMono-Regular.ttf", fBold="JetBrainsMono-SemiBold.ttf", sizes=(24, 19, 22, 20),
         tick=(660, 440), music=dict(kind="pluck", root=52, scale=[0, 1, 3, 5, 7, 8, 10], steps=64, step=.6, seed=53)),
]

def build(c):
    folder = f"{c['name']} by {AUTHOR}"
    root = os.path.join(OUT, folder); skin = os.path.join(root, "skin"); extra = os.path.join(skin, "extra"); snd = os.path.join(root, "sound")
    shutil.rmtree(root, ignore_errors=True)
    for p in (extra, snd): os.makedirs(p)
    def save(img, name, d=skin): img.save(os.path.join(d, name + ".png"), optimize=True)
    ft, fg = font(c["fTitle"], c["sizes"][0]), font(c["fGrid"], c["sizes"][1])

    bg = background(c); save(bg.convert("RGB"), "background")
    save(bar(c, 60, True), "bg-title"); save(bar(c, 60, False), "tips-bar-bg")
    icons = {}
    for i, f in enumerate(ICON_FILES):
        for st in ("n", "f"):
            icons[(i, st)] = main_icon(i, c, st == "f"); save(icons[(i, st)], f"ic-{f}-{st}")
    save(highlight(c, 640, 94), "bg-list-l"); save(highlight(c, 640, 60), "bg-list-s")
    save(panel(c, 640, 330, (44, 10, 596, 320)), "pop-bg")
    for k, h in ((1, 70), (2, 130), (3, 190)): save(panel(c, 320, h), f"bg-pop-menu-{k}")
    save(panel(c, 232, 68, hl=True), "bg-button-f")
    save(panel(c, 154, 170, (4, 4, 150, 166), fw=1), "bg-game-item-n"); save(panel(c, 154, 170, (4, 4, 150, 166), hl=True, fw=3), "bg-game-item-f")
    save(panel(c, 214, 120, fw=1), "bg-ra-list-item")
    save(panel(c, 256, 364, (6, 6, 250, 358), fw=2), "preview-bg")
    th = panel(c, 212, 304, (4, 4, 208, 300), fw=2)
    if c["kind"] == "landform": stipple(lf_icon_density(2, 48), 2, c["dim"], th, (58, 92))
    else: th.alpha_composite(keep_icon(3, c, False), (40, 86))
    text(th, (106, 236), "no image", fg, c["dim2"]); save(th, "thumb-default")
    e = new(208, 62); text(e, (104, 31), "empty", font(c["fTitle"], 34), c["dim2"]); save(e, "Empty")
    ln = new(90, 14); fn = font("JetBrainsMono-SemiBold.ttf", 13)
    for k in range(10): text(ln, (k * 9 + 4, 7), str(k), fn, c["dim2"])
    save(ln, "list-num")
    p = Pen(24, 14, c["accent"]); p.circle(.5, .5, .42); save(p.out(), "dot-a")
    p = Pen(24, 14, c["dim2"]); p.circle(.5, .5, .36, .12); save(p.out(), "dot-n")
    p = Pen(8, 8, c["accent"]); p.circle(.5, .5, .5); save(p.out(), "progress-dot")
    save(new(640, 4, rgb(c["dim"], 150)), "div-line-h"); save(new(4, 360, rgb(c["dim"], 150)), "div-line-v-01"); save(new(1, 60), "icon-game")
    logo = new(148, 42); text(logo, (0, 21), c["word"], font(c["fTitle"], 28), c["ink"], "lm"); save(logo, "miyoo-topbar")
    for lvl, nm in ((0, "power-0%-icon"), (.2, "power-20%-icon"), (.5, "power-50%-icon"), (.8, "power-80%-icon"), (1, "power-full-icon")):
        save(battery_small(c, lvl), nm)
    save(battery_small(c, 0, bolt=True), "ic-power-charge-100%")
    for k in range(1, 6):
        im = new(192, 96); im.alpha_composite(battery_big(c, 172, 76, k - 1), (10, 10)); save(im, f"charging-0{k}")
    fp = font(c["fBold"], 15)
    save(pill(c, "MENU", 105, 35, fp), "ic-MENU"); save(pill(c, "MENU+A", 124, 35, fp), "ic-MENU+A")
    save(button(c, "A", True), "icon-A-54"); save(button(c, "B", False), "icon-B-54")
    for nm, (w, h, fn_) in GLYPHS.items(): save(glyph(fn_, w, h, c["ink"]), nm)
    save(glyph(g_star, 34, 39, c["accent"] if c["kind"] == "keep" else c["dim2"]), "ic-favorite-mark")

    # extra
    save(plate_screen(c, c["word"]).convert("RGB"), "bootScreen", extra)
    save(plate_screen(c, "goodbye").convert("RGB"), "Screen_Off", extra)
    save(plate_screen(c, "saving", "your game is being saved").convert("RGB"), "Screen_Off_Save", extra)
    save(plate_screen(c, "battery low", "plug in to keep playing", art=battery_big(c, 120, 56, 1, fill=c["low"])).convert("RGB"), "lowBat", extra)
    save(bar(c, 40, True).convert("RGB"), "gs-top-bar", extra); save(bar(c, 40, False).convert("RGB"), "gs-bottom-bar", extra)
    for on in (0, 1):
        t = new(152, 32); frame(t, (120, 0, 152, 32), c["ink"], 2, fill=c["accent"] if on else None)
        if on:
            pc = Pen(32, 32, c["onAccent"]); pc.line([(.24, .52), (.43, .7), (.76, .32)], .11); t.alpha_composite(pc.out(), (120, 0))
        save(t, "toggle-on" if on else "toggle-off", extra)
    for k in range(11):
        l = new(300, 40); frame(l, (0, 4, 300, 36), c["ink"], 2, fill=c["bg"])
        if k: ImageDraw.Draw(l).rectangle((40, 10, 40 + round(254 * k / 10) - 1, 29), fill=rgb(c["accent"]))
        l.alpha_composite(glyph(g_brightness, 24, 24, c["ink"]), (9, 8)); save(l, f"lum{k}", extra)
    for k in range(FRAMES):
        fr = background(c, k / FRAMES, charging=True)
        frame(fr, (200, 170, 440, 310), c["dim"], 2, fill=c["bg"])
        fr.alpha_composite(battery_big(c, 150, 70, 1 + (k * 4 // FRAMES)), (245, 205))
        save(fr.convert("RGB"), f"chargingState{k}", extra)
    with open(os.path.join(extra, "chargingState.json"), "w") as f: f.write('{ "frame_delay": 110 }\n')

    # sound, fonts, config
    make_change(os.path.join(snd, "change.wav"), c); make_bgm(os.path.join(snd, "bgm.mp3"), c)
    used = sorted({c["fTitle"], c["fGrid"], c["fList"]})
    for f in used: shutil.copy(FONTS[f], os.path.join(root, f))
    s = c["sizes"]
    cfg = {"name": c["name"], "author": AUTHOR, "description": c["desc"],
           "batteryPercentage": {"visible": True, "color": c["ink"], "size": 18, "offsetX": 26, "fixed": True},
           "title": {"font": c["fTitle"], "size": s[0], "color": c["ink"]},
           "hint": {"font": c["fGrid"], "size": s[3], "color": c["ink"]},
           "currentpage": {"color": c["ink"]}, "total": {"color": c["dim2"]},
           "grid": {"font": c["fGrid"], "grid1x4": s[1], "grid3x4": max(16, s[1] - 5), "color": c["ink"], "selectedcolor": c["selInk"]},
           "list": {"font": c["fList"], "size": s[2], "color": c["ink"]}}
    with open(os.path.join(root, "config.json"), "w") as f: json.dump(cfg, f, indent=4); f.write("\n")

    # mock preview (the repo wants a real device screenshot; this is a stand-in)
    pv = bg.copy(); pv.alpha_composite(bar(c, 60, True)); pv.alpha_composite(bar(c, 60, False), (0, 420))
    pv.alpha_composite(logo, (20, 9)); pv.alpha_composite(battery_small(c, .8), (540, 6)); text(pv, (566, 30), "87%", font(c["fBold"], 18), c["ink"], "lm")
    for k in range(4):
        i = k + 1
        pv.alpha_composite(icons[(i, "f" if k == 1 else "n")], (8 + k * 156, 66))
        text(pv, (8 + k * 156 + 78, 66 + 221), LABELS[i], fg, c["selInk"] if k == 1 else c["ink"])
    fh = font(c["fGrid"], c["sizes"][3])
    pv.alpha_composite(button(c, "A", True), (20, 434)); text(pv, (60, 450), "SELECT", fh, c["ink"], "lm")
    pv.alpha_composite(button(c, "B", False), (190, 434)); text(pv, (230, 450), "BACK", fh, c["ink"], "lm")
    pv.save(os.path.join(root, "preview-full.png")); pv.resize((480, 360), Image.LANCZOS).save(os.path.join(root, "preview.png"))

    zpath = os.path.join(OUT, folder + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fns in os.walk(root):
            for fn_ in sorted(fns):
                if fn_ == "preview-full.png": continue
                full = os.path.join(dp, fn_); z.write(full, os.path.relpath(full, OUT))
    return root, zpath

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for c in THEMES:
        root, z = build(c)
        print(z, round(os.path.getsize(z) / 1024), "KB")
