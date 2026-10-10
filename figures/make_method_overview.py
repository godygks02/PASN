"""Method-overview figure, 1272 x 264 px.

(a) one global multi-basis bank: every basis runs for every input.
(b) exponent-bit router: the input's binade selects one bank; only that
    bank's bases run, with its own (N_j, T_j) budget.

Pure SVG string building. `--png` also rasterises to method_overview.png
(1272x264) and method_overview@2x.png (2544x528) with headless Chrome/Edge:

    python figures/make_method_overview.py --png
"""
import os
import shutil
import subprocess
import sys

W, H = 1272, 264
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# palette
INK = "#222222"
MUTED = "#6b6b6b"
AXIS = "#444444"
BAND_IDLE = "#f1f1f1"
BAND_SEL = "#fff1e4"
BAND_SEL_STROKE = "#f2b27d"
BANK_ON_FILL = "#e8effd"
BANK_ON_STROKE = "#2f6fed"
BANK_OFF_FILL = "#fafafa"
BANK_OFF_STROKE = "#c9c9c9"
BASIS_ON = "#2f6fed"
BASIS_OFF_STROKE = "#b9b9b9"
SPIKE = "#1a46c9"
BASE_ON = "#a9bdf2"
BASE_OFF = "#d6d6d6"
ROUTER_FILL = "#fde3cd"
ROUTER_STROKE = "#e07b28"
ROUTER_TEXT = "#a24e00"
MANT_FILL = "#dfe9fc"
MANT_STROKE = "#2f6fed"
MANT_TEXT = "#1a46c9"

out = []


def add(s):
    out.append(s)


def text(x, y, s, size=10, fill=INK, anchor="middle", italic=False, weight=None, family=FONT):
    style = f'font-family="{family}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"'
    if italic:
        style += ' font-style="italic"'
    if weight:
        style += f' font-weight="{weight}"'
    add(f'<text x="{x}" y="{y}" {style}>{s}</text>')


def line(x1, y1, x2, y2, stroke=INK, width=1, dash=None, cap="butt", marker=None):
    extra = ""
    if dash:
        extra += f' stroke-dasharray="{dash}"'
    if marker:
        extra += f' marker-end="url(#{marker})"'
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="{cap}"{extra}/>')


def rect(x, y, w, h, fill, stroke="none", width=1, rx=0, dash=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{extra}/>')


def circle(cx, cy, r, fill, stroke="none", width=1):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


# ---------------------------------------------------------------- bank glyph
ROW_H = 18
PAD = 8
R_BASIS = 5.5
DX = 8          # px per timestep
GAP_CT = 6      # circle -> train gap
TICK_H = 11

SPIKE_PATTERNS = [
    [0, 1, 3, 6],
    [0, 2, 5],
    [1, 3, 4, 7],
    [0, 1, 2, 5],
    [0, 3, 6],
    [1, 2, 4, 7],
]


def bank_width(T):
    return PAD + 2 * R_BASIS + GAP_CT + (T - 1) * DX + PAD


def bank(x, y, N, T, on, width=None):
    """Draw a bank whose top-left is (x, y). Returns (w, h, circle_cx)."""
    w = width or bank_width(T)
    h = PAD + N * ROW_H + PAD
    if on:
        rect(x, y, w, h, BANK_ON_FILL, BANK_ON_STROKE, 1.2, rx=4)
    else:
        rect(x, y, w, h, BANK_OFF_FILL, BANK_OFF_STROKE, 1, rx=4)
    cx = x + PAD + R_BASIS
    x_train = cx + R_BASIS + GAP_CT
    for i in range(N):
        cy = y + PAD + i * ROW_H + ROW_H / 2
        if on:
            circle(cx, cy, R_BASIS, BASIS_ON)
        else:
            circle(cx, cy, R_BASIS, "white", BASIS_OFF_STROKE, 1)
        y_base = cy + 4
        line(x_train, y_base, x_train + (T - 1) * DX, y_base, BASE_ON if on else BASE_OFF, 1)
        if on:
            for t in SPIKE_PATTERNS[i % len(SPIKE_PATTERNS)]:
                if t < T:
                    xt = x_train + t * DX
                    line(xt, y_base, xt, y_base - TICK_H, SPIKE, 1.6, cap="round")
    return w, h, cx


# ---------------------------------------------------------------- header
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>'
    '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
    '<marker id="arrb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{SPIKE}"/></marker>'
    '</defs>')
rect(0, 0, W, H, "white")

Y_AXIS = 54
Y_BAND_TOP = 41
Y_BANK_TOP = 82
Y_SUM = 242

# ================================================================ (a) MBE
band_l, band_r = 40, 450
ax0, ax1 = 58, 432
UNIT = (ax1 - ax0) / 4.0
X_INS = [-1.5, 0.375, 1.6]                    # three demo inputs
xs_dot = [ax0 + (v + 2) * UNIT for v in X_INS]

text(10, 15, '<tspan font-weight="bold">(a)</tspan> Multi-basis neuron (MBE)', 11.5, INK, "start")

# one range = whole axis
bank_w = bank_width(8)
bank_x = xs_dot[1] - bank_w / 2
bank_h = PAD + 6 * ROW_H + PAD
Y_BAND_BOT_A = Y_BANK_TOP + bank_h + 22
rect(band_l, Y_BAND_TOP, band_r - band_l, Y_BAND_BOT_A - Y_BAND_TOP, BAND_SEL, BAND_SEL_STROKE, 1, rx=5)

# axis
line(ax0, Y_AXIS, ax1, Y_AXIS, AXIS, 1)
for v, lab in [(-2, "−2"), (-1, None), (0, "0"), (1, None), (2, "2")]:
    xx = ax0 + (v + 2) * UNIT
    line(xx, Y_AXIS, xx, Y_AXIS + 4, AXIS, 1)
    if lab:
        text(xx, Y_AXIS + 14, lab, 9.5, MUTED)

# inputs: wherever x lies, the whole bank runs
land = [bank_x + 16, bank_x + bank_w / 2, bank_x + bank_w - 16]
for k, (xd, xl) in enumerate(zip(xs_dot, land)):
    circle(xd, Y_AXIS, 3.6, INK)
    text(xd + 6, Y_AXIS - 5, f'x<tspan baseline-shift="sub" font-size="7.5">{k + 1}</tspan>', 10.5, INK, "start", italic=True)
    y0, y1 = Y_AXIS + 5, Y_BANK_TOP - 2
    add(f'<path d="M{xd},{y0} C{xd},{y1} {xl},{y0} {xl},{y1}" fill="none" stroke="{INK}" stroke-width="1.2" marker-end="url(#arr)"/>')

# the single global bank, all bases on
w, h, cx = bank(bank_x, Y_BANK_TOP, 6, 8, True)
xc_bank = bank_x + w / 2
text(xc_bank + 8, Y_BANK_TOP + h + 14, "N × T", 10, INK, "start", italic=True)

# output
line(xc_bank, Y_BANK_TOP + h + 1, xc_bank, Y_SUM - 10, INK, 1.2, marker="arr")
circle(xc_bank, Y_SUM, 8.5, "white", INK, 1.2)
text(xc_bank, Y_SUM + 4, "Σ", 11, INK)
line(xc_bank + 9, Y_SUM, xc_bank + 24, Y_SUM, INK, 1.2, marker="arr")
text(xc_bank + 28, Y_SUM + 4, "f̂(x)", 11, INK, "start", italic=True)

# ================================================================ divider
line(482, 10, 482, H - 10, "#d8d8d8", 1)

# ================================================================ (b) ours
text(500, 15, '<tspan font-weight="bold">(b)</tspan> Ours', 11.5, INK, "start")

bx0 = 522
SLOT = 100
NSLOT = 7
SEL = 4                                          # x in [1/4, 1/2)
# per-bank budget (N_j, T_j)
BUDGET = [(1, 4), (1, 8), (2, 8), (3, 8), (2, 8), (2, 8), (1, 4)]
max_h = PAD + max(n for n, _ in BUDGET) * ROW_H + PAD
Y_BAND_BOT_B = Y_BANK_TOP + max_h + 22

# bands (binades)
for i in range(NSLOT):
    xl = bx0 + i * SLOT
    if i == SEL:
        rect(xl + 1, Y_BAND_TOP, SLOT - 2, Y_BAND_BOT_B - Y_BAND_TOP, BAND_SEL, BAND_SEL_STROKE, 1, rx=5)
    else:
        rect(xl + 1, Y_BAND_TOP, SLOT - 2, Y_BAND_BOT_B - Y_BAND_TOP, BAND_IDLE, rx=5)

# axis with binade boundaries
line(bx0, Y_AXIS, bx0 + NSLOT * SLOT, Y_AXIS, AXIS, 1)
labels = ["−2", "−1", "−½", "−¼", "¼", "½", "1", "2"]
for i, lab in enumerate(labels):
    xx = bx0 + i * SLOT
    line(xx, Y_AXIS, xx, Y_AXIS + 4, AXIS, 1)
    text(xx, Y_AXIS + 14, lab, 9.5, MUTED)
x_zero = bx0 + 3 * SLOT + SLOT / 2
line(x_zero, Y_AXIS, x_zero, Y_AXIS + 4, AXIS, 1)
text(x_zero, Y_AXIS + 14, "0", 9.5, MUTED)

# banks
bank_w_b = bank_width(8)
sel_cx = None
for i, (n, t) in enumerate(BUDGET):
    xl = bx0 + i * SLOT + (SLOT - bank_w_b) / 2
    w, h, cx = bank(xl, Y_BANK_TOP, n, t, i == SEL, width=bank_w_b)
    if i == SEL:
        sel_cx, sel_h = xl + w / 2, h

# input: bit field above the selected band, callout to the dot on the axis
x_dot_b = sel_cx
circle(x_dot_b, Y_AXIS, 3.6, INK)

CW, CH = 11, 13
y_cells = 19
sign_bits = ["0"]
exp_bits = list("01111101")          # 125 -> 2^-2
man_bits = list("10000")             # x = 0.375 = 1.5 * 2^-2  ->  rho = 0.5
groups = [(sign_bits, ROUTER_FILL, ROUTER_STROKE, ROUTER_TEXT),
          (exp_bits, ROUTER_FILL, ROUTER_STROKE, ROUTER_TEXT),
          (man_bits, MANT_FILL, MANT_STROKE, MANT_TEXT)]
total = sum(len(g[0]) * CW for g in groups) + 3 * (len(groups) - 1)
xf = x_dot_b - total / 2 + 12
text(xf - 6, y_cells + CH - 3, "x =", 10.5, INK, "end", italic=True)
xc = xf
group_x = []
for bits, fill, stroke, tcol in groups:
    gx0 = xc
    for b in bits:
        rect(xc, y_cells, CW, CH, fill, stroke, 0.8)
        text(xc + CW / 2, y_cells + CH - 3, b, 8.5, tcol, family="Consolas, Menlo, monospace")
        xc += CW
    group_x.append((gx0, xc))
    xc += 3
text(xc + 1, y_cells + CH - 3, "…", 9, MANT_TEXT, "start")
# group labels
text((group_x[0][0] + group_x[0][1]) / 2, y_cells - 4, "sign", 8, ROUTER_TEXT)
text((group_x[1][0] + group_x[1][1]) / 2, y_cells - 4, "exponent", 8, ROUTER_TEXT)
text((group_x[2][0] + group_x[2][1]) / 2, y_cells - 4, "mantissa", 8, MANT_TEXT)
# router pointer: sign+exponent bits pick the band
xr = (group_x[0][0] + group_x[1][1]) / 2
add(f'<path d="M{xr - 4},{y_cells + CH + 1} L{xr + 4},{y_cells + CH + 1} L{xr},{Y_BAND_TOP + 1} z" fill="{ROUTER_STROKE}"/>')
# callout: bits -> dot
line(x_dot_b, y_cells + CH + 1, x_dot_b, Y_AXIS - 5, "#9a9a9a", 0.8)

# mantissa residual into the selected bank
line(x_dot_b, Y_AXIS + 5, x_dot_b, Y_BANK_TOP - 2, SPIKE, 1.3, marker="arrb")
text(x_dot_b + 6, Y_BANK_TOP - 6, "ρ", 10.5, MANT_TEXT, "start", italic=True)

# per-bank budget label under the selected bank
text(x_dot_b + 8, Y_BANK_TOP + sel_h + 14, "N<tspan baseline-shift=\"sub\" font-size=\"7.5\">j</tspan> × T<tspan baseline-shift=\"sub\" font-size=\"7.5\">j</tspan>", 10, INK, "start", italic=True)

# output from the selected bank only
line(x_dot_b, Y_BANK_TOP + sel_h + 1, x_dot_b, Y_SUM - 10, INK, 1.2, marker="arr")
circle(x_dot_b, Y_SUM, 8.5, "white", INK, 1.2)
text(x_dot_b, Y_SUM + 4, "Σ", 11, INK)
line(x_dot_b + 9, Y_SUM, x_dot_b + 24, Y_SUM, INK, 1.2, marker="arr")
text(x_dot_b + 28, Y_SUM + 4, "f̂(x)", 11, INK, "start", italic=True)

add('</svg>')

svg = "\n".join(out)
here = os.path.dirname(os.path.abspath(__file__))
stem = os.path.join(here, "method_overview")
with open(stem + ".svg", "w", encoding="utf-8") as f:
    f.write(svg)
print("wrote", stem + ".svg")

if "--png" in sys.argv:
    html = stem + ".render.html"
    with open(html, "w", encoding="utf-8") as f:
        f.write('<!doctype html><meta charset="utf-8">'
                '<body style="margin:0;overflow:hidden;background:#fff">' + svg + '</body>')
    browser = next((b for b in [
        shutil.which("chrome"), shutil.which("msedge"),
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    ] if b and os.path.exists(b)), None)
    if browser is None:
        sys.exit("no Chrome/Edge found; open the .svg or render it another way")
    for scale, suffix in [(1, ".png"), (2, "@2x.png")]:
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--force-device-scale-factor={scale}", f"--window-size={W},{H}",
                        f"--screenshot={stem}{suffix}", "file:///" + html.replace(os.sep, "/")],
                       check=True, capture_output=True)
        print("wrote", stem + suffix)
    os.remove(html)
