"""Draw the two diagrams in docs/img/: well_as_image.svg and two_stage_unet.svg.

    python scripts/make_diagrams.py docs/img

The well in them is a toy (64 x 128, made up here), not a competition well. The reference log, the path and
the well's log are built from each other, so the pictures agree. Colours switch with the viewer's dark mode.
"""
import math
import sys
from pathlib import Path

import base64
import struct
import zlib

import numpy as np

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

STYLE = """<style>
  .bg{fill:#f8faf9} .ink{fill:#16222b} .mut{fill:#55646e}
  .box{fill:#eef2f0;stroke:#c6d1ce;stroke-width:1.5}
  .kbox{fill:#d3e7ee;stroke:#1b6a86;stroke-width:1.5}
  .lbox{fill:#f4d9df;stroke:#a3304a;stroke-width:1.5}
  .frame{fill:#ffffff;stroke:#c6d1ce;stroke-width:1.5}
  .ar{stroke:#55646e;stroke-width:1.8;fill:none;marker-end:url(#a)}
  .arl{stroke:#55646e;stroke-width:1.8;fill:none}
  .dash{stroke:#a3304a;stroke-width:2.4;stroke-dasharray:7 5;fill:none}
  .known{stroke:#16222b;stroke-width:3;fill:none}
  .lrule{stroke:#55646e;stroke-width:1.3;fill:none}
  .rule{stroke:#c6d1ce;stroke-width:1}
  .head{fill:#55646e} .kt{fill:#1b6a86} .lt{fill:#a3304a}
  .sw{fill:#1b6a86} .ly{fill:#b07d3c} .dot{fill:#16222b} .pin{fill:#a3304a}
  .prob{stroke:#1b6a86;fill:none;stroke-linecap:round}
  @media (prefers-color-scheme: dark){
    .bg{fill:#172329} .ink{fill:#e7eef1} .mut{fill:#9db0b9}
    .box{fill:#111a1f;stroke:#2c3c44} .kbox{fill:#15323d;stroke:#73c4de} .lbox{fill:#3d1c26;stroke:#f08aa0}
    .frame{fill:#0d1418;stroke:#2c3c44}
    .ar{stroke:#9db0b9} .arl{stroke:#9db0b9} .dash{stroke:#f08aa0} .known{stroke:#e7eef1} .lrule{stroke:#9db0b9}
    .rule{stroke:#2c3c44} .head{fill:#9db0b9} .kt{fill:#73c4de} .lt{fill:#f08aa0}
    .sw{fill:#73c4de} .ly{fill:#d9a560} .dot{fill:#e7eef1} .pin{fill:#f08aa0} .prob{stroke:#73c4de}
  }
  text{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:15px}
  .h{font-weight:600;font-size:16px} .s{font-size:13px} .step{font-weight:700;font-size:15px}
</style>
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" class="head"/></marker></defs>"""


def head(w, h, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d">',
            f'<title id="t">{title}</title>', f'<desc id="d">{desc}</desc>', STYLE,
            f'<rect width="{w}" height="{h}" rx="14" class="bg"/>']


def f(x):
    return f'{x:.1f}'.rstrip('0').rstrip('.')


def poly(pts):
    return 'M' + ' L'.join(f'{f(x)} {f(y)}' for x, y in pts)


def text(x, y, s, cls='ink', anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ''
    return f'<text x="{f(x)}" y="{f(y)}"{a} class="{cls}">{s}</text>'


# ---------------------------------------------------------------- shared toy well
# A reference log (GR against depth in the layers) and a path through it. The well's own
# log is the reference log read along the path, so every picture below agrees with the others.
rng = np.random.default_rng(7)
N_ROWS, N_COLS = 64, 128                 # toy resolution (the real image is 256 x 512)


def tw_gr(r):
    """Reference GR at fractional row r (0 = top): a few beds of different brightness."""
    beds = [(0, .25), (7, .55), (12, .35), (19, .85), (24, .5), (31, .2), (36, .7), (42, .45),
            (48, .9), (53, .3), (58, .6), (64, .4)]
    v = beds[0][1]
    for top, val in beds:
        if r >= top:
            v = val
    return v + 0.06 * math.sin(r * 1.7)


def path_row(c):
    """True row per column: flat-ish at the heel, then drifts down and back up toward the toe."""
    x = c / (N_COLS - 1)
    return 30 + 9 * math.sin(2.4 * x) + 6 * math.sin(7.5 * x + 0.6) * x


PS = int(0.38 * N_COLS)                    # the PS point (known to the left of it)
truth = np.array([path_row(c) for c in range(N_COLS)])
hw = np.array([tw_gr(r) for r in truth]) + rng.normal(0, 0.04, N_COLS)
twv = np.array([tw_gr(r + 0.5) for r in range(N_ROWS)])
dz = np.convolve(np.gradient(truth), np.ones(5) / 5, mode='same')


def stripes_rows(x, y, w, h, vals, op_lo=0.05, op_hi=0.75):
    """Horizontal bands: one value per row."""
    out, n = [], len(vals)
    for i, v in enumerate(vals):
        op = op_lo + (op_hi - op_lo) * v
        out.append(f'<rect x="{f(x)}" y="{f(y + i * h / n)}" width="{f(w)}" height="{f(h / n + .4)}" '
                   f'class="sw" fill-opacity="{op:.2f}"/>')
    return out


def stripes_cols(x, y, w, h, vals, op_lo=0.05, op_hi=0.75):
    out, n = [], len(vals)
    for i, v in enumerate(vals):
        op = op_lo + (op_hi - op_lo) * v
        out.append(f'<rect x="{f(x + i * w / n)}" y="{f(y)}" width="{f(w / n + .4)}" height="{f(h)}" '
                   f'class="sw" fill-opacity="{op:.2f}"/>')
    return out


def png_gray(a):
    """Grayscale PNG (as a data URI) from a 2-D array in [0, 1]."""
    a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    h, w = a.shape
    raw = b''.join(b'\x00' + a[i].tobytes() for i in range(h))

    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) \
        + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode()


def img_xy(x0, y0, w, h, c, r):
    return x0 + (c + 0.5) * w / N_COLS, y0 + r * h / N_ROWS


# ================================================================ figure 1: the well as an image
W1, H1 = 1000, 640
s = head(W1, H1, 'A horizontal well drawn as an image',
         'Left: a cross-section of the ground. A vertical typewell gives the reference gamma-ray log, layer by layer. '
         'A horizontal well runs sideways through layers that bend and dip; its position in the layers is known up to '
         'the PS point and must be predicted after it. Right: the same well drawn as an image. Each column is one step '
         'along the well; each row is one foot of depth in the layers, so the layers become flat horizontal bands. The '
         'well becomes a line wandering across the bands: dots where it is known, a dashed line where it must be '
         'predicted. The reference log runs down the left side and the well\'s own log along the bottom. Below: the '
         'five channels the network reads: the reference log, the well\'s log, the known positions, a matching cost '
         'that is dark where the two logs agree, and the well\'s dip. Predicting the depth becomes picking one row '
         'in every column.')

s.append(text(24, 36, 'IN THE GROUND', 'step kt'))
s.append(text(158, 36, 'What a geologist looks at.', 'ink'))
s.append(text(500, 36, 'AS AN IMAGE', 'step kt'))
s.append(text(612, 36, 'The same well, with the layers flattened.', 'ink'))

# ---- left: cross-section
L, T, R, B = 24, 56, 430, 400
s.append(f'<rect x="{L}" y="{T}" width="{R - L}" height="{B - T}" rx="10" class="frame"/>')
s.append(f'<clipPath id="cx"><rect x="{L}" y="{T}" width="{R - L}" height="{B - T}" rx="10"/></clipPath>')
s.append('<g clip-path="url(#cx)">')


def surf(x, base):
    """Depth of a layer boundary at x: dips gently and bends, the same shape for every layer."""
    u = (x - L) / (R - L)
    return base - 70 * u + 18 * math.sin(5.5 * u + 0.4) + 10 * math.sin(11 * u)


bounds = list(range(40, 560, 26))
ops = [0.10, 0.32, 0.18, 0.45, 0.24, 0.12, 0.38, 0.2, 0.5, 0.15, 0.3, 0.42, 0.22, 0.12, 0.36, 0.28, 0.18, 0.4, 0.25, 0.33]
xs = np.linspace(L, R, 80)
for i in range(len(bounds) - 1):
    top = [(x, surf(x, bounds[i])) for x in xs]
    bot = [(x, surf(x, bounds[i + 1])) for x in xs[::-1]]
    s.append(f'<path d="{poly(top + bot)} Z" class="ly" fill-opacity="{ops[i % len(ops)]:.2f}"/>')

# typewell: a vertical line with its GR log wiggle beside it
twx = 46
s.append(f'<path d="M{twx} {T} V{B}" class="known"/>')
wig = []
for yy in np.linspace(T + 8, B - 8, 160):
    # brightness of the layer at the typewell
    k = 0
    for i in range(len(bounds) - 1):
        if surf(twx, bounds[i]) <= yy < surf(twx, bounds[i + 1]):
            k = i
    wig.append((twx + 8 + 30 * ops[k % len(ops)] / 0.5, yy))
s.append(f'<path d="{poly(wig)}" class="lrule"/>')

# horizontal well: comes down, turns, then runs sideways, staying near the target layer
def well_y(x):
    u = (x - 250) / (R - 250)
    return 245 - 70 * u + 18 * math.sin(5.5 * ((x - L) / (R - L)) + 0.4) * 0.55 + 8 * u


heel = 250
yh = well_y(heel)
s_top = f'M196 {T} C196 {f(yh - 30)} {heel - 40} {f(yh + 8)} {heel} {f(yh)}'
curve = []
ps_x = 330
known = [(x, well_y(x)) for x in np.linspace(heel, ps_x, 40)]
todo = [(x, well_y(x)) for x in np.linspace(ps_x, R - 14, 50)]
s.append(f'<path d="{s_top}" class="known"/>')
s.append(f'<path d="{poly(known)}" class="known"/>')
s.append(f'<path d="{poly(todo)}" class="dash"/>')
s.append('</g>')
px, py = ps_x, well_y(ps_x)
s.append(f'<circle cx="{f(px)}" cy="{f(py)}" r="5.5" class="pin"/>')

s.append(text(twx + 46, T + 22, 'typewell', 's ink'))
s.append(text(twx + 46, T + 38, '(reference log)', 's mut'))
s.append(text(heel - 20, well_y(heel) + 34, 'known position', 's ink'))
s.append(text(px - 8, py - 16, 'PS point', 's lt'))
s.append(text(R - 16, well_y(R - 16) + 32, 'predict this', 's lt', 'end'))
s.append(text(L + 74, B - 14, 'layers dip and bend; the well crosses them', 's mut'))

# arrow between panels
s.append(f'<path d="M{R + 6} 228 H{R + 34}" class="ar"/>')

# ---- right: the image
X0, Y0, IW, IH = 524, 72, 384, 280
s.extend(stripes_rows(X0, Y0, IW, IH, twv, 0.04, 0.55))
s.append(f'<rect x="{X0}" y="{Y0}" width="{IW}" height="{IH}" fill="none" class="lrule"/>')
# known positions: dots; to predict: dashed line
for c in range(0, PS, 3):
    x, y = img_xy(X0, Y0, IW, IH, c, truth[c])
    s.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="2.6" class="dot"/>')
pts = [img_xy(X0, Y0, IW, IH, c, truth[c]) for c in range(PS, N_COLS)]
s.append(f'<path d="{poly(pts)}" class="dash"/>')
x_ps = X0 + PS * IW / N_COLS
s.append(f'<path d="M{f(x_ps)} {Y0} V{Y0 + IH}" class="lrule" stroke-dasharray="3 4"/>')
s.append(f'<rect x="{f(x_ps + 2)}" y="{Y0 + IH - 24}" width="62" height="20" rx="4" class="bg" fill-opacity="0.85"/>')
s.append(text(x_ps + 6, Y0 + IH - 9, 'PS point', 's lt'))
s.append(text(X0 + 8, Y0 + 16, 'known', 's ink'))
s.append(text(X0 + IW - 8, Y0 + 16, 'predict: one row per column', 's lt', 'end'))
# reference log down the left side
wig = [(X0 - 8 - 30 * twv[r], Y0 + (r + .5) * IH / N_ROWS) for r in range(N_ROWS)]
s.append(f'<path d="{poly(wig)}" class="lrule"/>')
s.append(f'<text transform="translate({X0 - 46} {Y0 + IH / 2}) rotate(-90)" text-anchor="middle" class="s mut">reference log, down the rows</text>')
# the well's log along the bottom
wig = [(X0 + (c + .5) * IW / N_COLS, Y0 + IH + 44 - 30 * hw[c]) for c in range(N_COLS)]
s.append(f'<path d="{poly(wig)}" class="lrule"/>')
s.append(text(X0 + IW / 2, Y0 + IH + 66, "the well's own log, across the columns", 's mut', 'middle'))
# axis labels
s.append(text(X0 + IW + 12, Y0 + 10, 'rows:', 's ink'))
s.append(text(X0 + IW + 12, Y0 + 26, '1 ft of', 's mut'))
s.append(text(X0 + IW + 12, Y0 + 42, 'depth', 's mut'))
s.append(text(X0 + IW + 12, Y0 + 58, '(256)', 's mut'))
s.append(text(X0 + IW + 12, Y0 + IH - 36, 'columns:', 's ink'))
s.append(text(X0 + IW + 12, Y0 + IH - 20, 'along the', 's mut'))
s.append(text(X0 + IW + 12, Y0 + IH - 4, 'well (512)', 's mut'))

s.append(f'<line x1="24" y1="452" x2="976" y2="452" class="rule"/>')
s.append(text(24, 482, 'FIVE CHANNELS', 'step kt'))
s.append(text(170, 482, 'What the network reads, stacked like the red, green and blue of a photo.', 'ink'))

# ---- channel tiles
tiles = ['reference log', "well's log", 'known rows', 'match cost', 'dip']
subs = ['by depth', 'by step along well', 'pre-PS positions', 'marks where logs agree', 'per column']
TW_, TH_, TY = 168, 84, 500
gap = (952 - 5 * TW_) / 4
for i, (name, sub) in enumerate(zip(tiles, subs)):
    tx = 24 + i * (TW_ + gap)
    s.append(f'<rect x="{f(tx)}" y="{TY}" width="{TW_}" height="{TH_}" class="frame"/>')
    if i == 0:
        s.extend(stripes_rows(tx, TY, TW_, TH_, twv, .04, .7))
    elif i == 1:
        s.extend(stripes_cols(tx, TY, TW_, TH_, hw, .04, .7))
    elif i == 2:
        for c in range(0, PS, 3):
            x = tx + (c + .5) * TW_ / N_COLS
            y = TY + truth[c] * TH_ / N_ROWS
            s.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="1.8" class="dot"/>')
    elif i == 3:
        cost = np.array([[abs(hw[c] - tw_gr(r + .5)) for c in range(N_COLS)] for r in range(N_ROWS)])
        lum = np.exp(-(cost / 0.09) ** 2)
        s.append(f'<mask id="mc" maskUnits="userSpaceOnUse" x="{f(tx)}" y="{TY}" width="{TW_}" height="{TH_}">'
                 f'<image href="{png_gray(lum)}" x="{f(tx)}" y="{TY}" width="{TW_}" height="{TH_}" '
                 f'preserveAspectRatio="none"/></mask>')
        s.append(f'<rect x="{f(tx)}" y="{TY}" width="{TW_}" height="{TH_}" class="sw" mask="url(#mc)"/>')
    else:
        v = (dz - dz.min()) / (dz.max() - dz.min())
        s.extend(stripes_cols(tx, TY, TW_, TH_, v, .04, .7))
    s.append(f'<rect x="{f(tx)}" y="{TY}" width="{TW_}" height="{TH_}" fill="none" class="lrule"/>')
    s.append(text(tx + TW_ / 2, TY + TH_ + 22, name, 'h ink', 'middle'))
    s.append(text(tx + TW_ / 2, TY + TH_ + 40, sub, 's mut', 'middle'))

s.append('</svg>')
(OUT / 'well_as_image.svg').write_text('\n'.join(s) + '\n')


# ================================================================ figure 2: the two-stage U-Net
def unet_icon(x, y, w, h, cls='kbox'):
    """A small U: encoder blocks going down, decoder blocks coming up, skip lines across."""
    out = [f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" rx="10" class="{cls}"/>']
    n, bw = 4, 13
    pad = 16
    ys = [y + 46 + i * (h - 70) / (n - 1) / 1.0 for i in range(n)]
    for i in range(n):
        hh = 34 - 6 * i
        xl = x + pad + i * 15
        xr = x + w - pad - bw - i * 15
        out.append(f'<rect x="{f(xl)}" y="{f(ys[i] - hh / 2)}" width="{bw}" height="{f(hh)}" rx="2" class="box"/>')
        out.append(f'<rect x="{f(xr)}" y="{f(ys[i] - hh / 2)}" width="{bw}" height="{f(hh)}" rx="2" class="box"/>')
        if i < n - 1:
            out.append(f'<path d="M{f(xl + bw + 2)} {f(ys[i])} H{f(xr - 2)}" class="lrule" stroke-dasharray="2 3"/>')
    return out


def prob_icon(x, y, w, h, sharp):
    """Row probabilities: a glowing band along the path; wider when less sure."""
    out = [f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" rx="6" class="frame"/>']
    pts = [(x + (c + .5) * w / N_COLS, y + (truth[c] + (0 if sharp else 3 * math.sin(c / 9))) * h / N_ROWS)
           for c in range(0, N_COLS, 2)]
    widths = [(3, .9), (8, .35), (16, .15)] if sharp else [(6, .55), (16, .25), (30, .12)]
    for sw, op in widths[::-1]:
        out.append(f'<path d="{poly(pts)}" class="prob" stroke-width="{sw}" stroke-opacity="{op}"/>')
    return out


W2, H2 = 1000, 572
s = head(W2, H2, 'The two-stage U-Net and how it is trained',
         'Top: the five-channel image goes into the stage 1 U-Net. Both U-Nets have EfficientNet encoders that start from ImageNet weights. Stage 1 '
         'gives, for every column, a probability for every row. These are softened (temperature 1.5) and added to the '
         'image as a sixth channel. Stage 2, a second U-Net, reads all six and gives sharper probabilities. Its output '
         'is fed back into itself two more times (three rounds; the L2 model runs one). The prediction in each column is the '
         'probability-weighted average row, turned back into depth in feet. Bottom: each model first learns on 25,000 '
         'synthetic wells per fold for 5 epochs, then on about 620 real wells for up to 35 epochs. Four models of '
         'different sizes, each trained 35 times (5 folds by 7 seeds), are averaged with equal weights.')

s.append(text(24, 36, 'THE MODEL', 'step kt'))
s.append(text(130, 36, 'Two U-Nets in a row. The second one checks and sharpens the first.', 'ink'))

# input image
ix, iy, iw, ih = 24, 94, 120, 84
s.extend(stripes_rows(ix, iy, iw, ih, twv, .04, .55))
for c in range(0, PS, 4):
    x, y = img_xy(ix, iy, iw, ih, c, truth[c])
    s.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="1.6" class="dot"/>')
s.append(f'<rect x="{ix}" y="{iy}" width="{iw}" height="{ih}" fill="none" class="lrule"/>')
s.append(text(ix + iw / 2, iy + ih + 22, 'the image', 'h ink', 'middle'))
s.append(text(ix + iw / 2, iy + ih + 40, '5 channels', 's mut', 'middle'))

# stage 1
s1x, s1y, uw, uh = 178, 80, 170, 128
s.append(f'<path d="M{ix + iw + 4} 136 H{s1x - 2}" class="ar"/>')
s.extend(unet_icon(s1x, s1y, uw, uh))
s.append(text(s1x + uw / 2, s1y + 26, 'Stage 1 U-Net', 'h ink', 'middle'))
s.append(text(s1x + uw / 2, s1y + uh + 22, 'EfficientNet encoder from', 's mut', 'middle'))
s.append(text(s1x + uw / 2, s1y + uh + 38, 'ImageNet (both stages)', 's mut', 'middle'))

# stage-1 probabilities
p1x, p1y, pw, ph = 382, 94, 120, 84
s.append(f'<path d="M{s1x + uw + 2} 136 H{p1x - 2}" class="ar"/>')
s.extend(prob_icon(p1x, p1y, pw, ph, sharp=False))
s.append(text(p1x + pw / 2, p1y + ph + 22, 'first guess', 'h ink', 'middle'))
s.append(text(p1x + pw / 2, p1y + ph + 40, 'chance of each row', 's mut', 'middle'))
s.append(text(p1x + pw / 2, p1y + ph + 56, 'softened (T = 1.5)', 's mut', 'middle'))

# stage 2 (reads the image too)
s2x = 556
s.append(f'<path d="M{p1x + pw + 2} 136 H{s2x - 2}" class="ar"/>')
s.append(text((p1x + pw + s2x) / 2, 130, '+', 'h ink', 'middle'))
s.extend(unet_icon(s2x, s1y, uw, uh))
s.append(text(s2x + uw / 2, s1y + 26, 'Stage 2 U-Net', 'h ink', 'middle'))
# the image also feeds stage 2: a line under stage 1
s.append(f'<path d="M84 {iy + ih + 50} V272 H{s2x + uw / 2} V{s1y + uh + 2}" class="ar" stroke-dasharray="4 4"/>')
s.append(text(s2x + uw / 2 + 12, 254, 'stage 2 also reads the 5 channels', 's mut'))

# stage-2 probabilities
p2x = 760
s.append(f'<path d="M{s2x + uw + 2} 136 H{p2x - 2}" class="ar"/>')
s.extend(prob_icon(p2x, p1y, pw, ph, sharp=True))
s.append(text(p2x + pw / 2, p1y + ph + 22, 'sharper guess', 'h ink', 'middle'))
# refeed loop
s.append(f'<path d="M{p2x + pw / 2} {p1y - 2} V64 H{s2x + uw / 2} V{s1y - 2}" class="ar"/>')
s.append(text((p2x + pw / 2 + s2x + uw / 2) / 2, 58, 'feed back in: 3 rounds in total', 's kt', 'middle'))

# decode
dx, dy = 914, 106
s.append(f'<path d="M{p2x + pw + 2} 136 H{dx - 2}" class="ar"/>')
s.append(f'<rect x="{dx}" y="{dy}" width="62" height="60" rx="30" class="lbox"/>')
s.append(text(dx + 31, dy + 27, 'TVT', 'h lt', 'middle'))
s.append(text(dx + 31, dy + 45, 'in ft', 's lt', 'middle'))
s.append(text(dx + 31, p1y + ph + 22, 'average', 's mut', 'middle'))
s.append(text(dx + 31, p1y + ph + 38, 'row, in feet', 's mut', 'middle'))

s.append(f'<line x1="24" y1="306" x2="976" y2="306" class="rule"/>')
s.append(text(24, 340, 'TRAINING', 'step kt'))
s.append(text(124, 340, 'Learn on made-up wells first, where the answer is always known.', 'ink'))

bx, by, bw_, bh_ = 24, 364, 286, 92
boxes = [
    ('box', '25,000 synthetic wells', ['per fold, from the reference logs:', 'faults, dip breaks, repeats', '5 epochs']),
    ('box', 'about 620 real wells', ['per fold; flips, stretches,', 'hidden hints; EMA weights', 'up to 35 epochs']),
    ('kbox', '4 models, averaged', ['EfficientNetV2-L (2 sizes), V2-XL, L2', 'each 5 folds × 7 seeds', 'equal weights, CV 5.038']),
]
gx = (952 - 3 * bw_) / 2
for i, (cls, h1, lines) in enumerate(boxes):
    x = bx + i * (bw_ + gx)
    s.append(f'<rect x="{f(x)}" y="{by}" width="{bw_}" height="{bh_ + 28}" rx="10" class="{cls}"/>')
    s.append(text(x + bw_ / 2, by + 28, h1, 'h ink', 'middle'))
    for j, ln in enumerate(lines):
        s.append(text(x + bw_ / 2, by + 52 + 20 * j, ln, 's mut', 'middle'))
    if i < 2:
        s.append(f'<path d="M{f(x + bw_ + 2)} {by + 60} H{f(x + bw_ + gx - 2)}" class="ar"/>')

s.append(text(24, 528, 'Only the rows inside the image are trained on. Both stages get a loss; stage 1 does not get gradients through the', 's mut'))
s.append(text(24, 546, 'hand-off. Stage 2 is trained to improve its own output (2 to 4 rounds on half the steps), so 3 rounds help at prediction time.', 's mut'))
s.append('</svg>')
(OUT / 'two_stage_unet.svg').write_text('\n'.join(s) + '\n')
print('ok', [p.stat().st_size for p in OUT.glob('*.svg')])
