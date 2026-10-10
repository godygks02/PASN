"""Bank-neuron figure in the layout of the MBE paper's Fig. 4.

Left panel: the neuron — input, exponent-bit router, one selected bank out of R,
output.  Right panel: inside the selected bank — the binade it owns, the
mantissa residual it receives, its own N_j bases run for T_j steps, readout.

Coordinates are in the 986 x 466 space of a 4x raster of the original figure
(outer frame 951 x 430), so the two figures overlay one-to-one.

    python figures/make_bank_neuron.py --png

writes bank_neuron.svg, bank_neuron.png (453 x 214 — the size of the reference
crop), bank_neuron@2x.png (986 x 466) and bank_neuron@4x.png (1972 x 932).
"""
import os
import shutil
import subprocess
import sys

W, H = 986, 466
SERIF = "Times New Roman, Times, STIX Two Text, serif"

INK = "#1f1f1f"
GRAY = "#8a8a8a"
BOX_BLUE = "#dbe7f4"       # basis/bank row box
BOX_BLUE_STROKE = "#9fb3c8"
LAV = "#ebe6f4"            # dashed parameter boxes
LAV_STROKE = "#8f8aa3"
ROUTER_FILL = "#fde3cd"
ROUTER_STROKE = "#e07b28"
ROUTER_TEXT = "#a24e00"
MANT_FILL = "#dfe9fc"
MANT_STROKE = "#2f6fed"
MANT_TEXT = "#1a46c9"
SEL_FILL = "#e8effd"
SEL_STROKE = "#2f6fed"
BASIS_ON = "#2f6fed"

out = []
add = out.append


def text(x, y, s, size=20, fill=INK, anchor="middle", italic=False, weight=None, family=SERIF):
    st = f'font-family="{family}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"'
    if italic:
        st += ' font-style="italic"'
    if weight:
        st += f' font-weight="{weight}"'
    add(f'<text x="{x}" y="{y}" {st}>{s}</text>')


def line(x1, y1, x2, y2, stroke=INK, width=2, dash=None, marker=None):
    ex = f' stroke-dasharray="{dash}"' if dash else ""
    ex += f' marker-end="url(#{marker})"' if marker else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{ex}/>')


def rect(x, y, w, h, fill, stroke="none", width=1, rx=0, dash=None):
    ex = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{ex}/>')


def circle(cx, cy, r, fill, stroke="none", width=1, dash=None):
    ex = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{ex}/>')


def sub(s, size=0.62):
    return f'<tspan baseline-shift="sub" font-size="{size}em">{s}</tspan>'


def sup(s, size=0.62):
    return f'<tspan baseline-shift="super" font-size="{size}em">{s}</tspan>'


def it(s):
    return f'<tspan font-style="italic">{s}</tspan>'


def subsup(lo, hi, size=0.62, back=0.32):
    """Subscript and superscript stacked at the same x (sub first, sup pulled back)."""
    return (f'<tspan baseline-shift="sub" font-size="{size}em">{lo}</tspan>'
            f'<tspan baseline-shift="super" font-size="{size}em" dx="-{back}em">{hi}</tspan>')


def rm(s):
    return f'<tspan font-style="normal">{s}</tspan>'


def loop_arrow(cx, cy, r, stroke=INK, width=2.8):
    """Three-quarter circular arrow (the 'T' loop of the original)."""
    import math
    a0, a1 = math.radians(300), math.radians(-30 + 360)  # start / end angles
    x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
    add(f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 1 0 {x1:.1f},{y1:.1f}" fill="none" '
        f'stroke="{stroke}" stroke-width="{width}" marker-end="url(#arr)"/>')


def sigma(x, y, top, bottom, size=42):
    """Big sigma with limits, baseline at y."""
    text(x, y, "∑", size, INK, family="Cambria Math, STIX Two Math, Times New Roman, serif")
    text(x, y - size * 0.86, top, 15, INK)
    text(x, y + 17, bottom, 15, INK)


# ------------------------------------------------------------------ canvas
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
add('<defs>'
    '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#eef2f8"/><stop offset="0.55" stop-color="#f8f8fa"/>'
    '<stop offset="1" stop-color="#f9f3ea"/></linearGradient>'
    '<linearGradient id="inner" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#f1f4f7"/></linearGradient>'
    '<radialGradient id="beige" cx="0.4" cy="0.35" r="0.8">'
    '<stop offset="0" stop-color="#fbf6ee"/><stop offset="1" stop-color="#ecd9bf"/></radialGradient>'
    '<radialGradient id="green" cx="0.4" cy="0.35" r="0.8">'
    '<stop offset="0" stop-color="#f1f6ec"/><stop offset="1" stop-color="#c5dcc9"/></radialGradient>'
    '<linearGradient id="wedge" x1="0" y1="0" x2="1" y2="0">'
    '<stop offset="0" stop-color="#b7d9ad" stop-opacity="0.75"/>'
    '<stop offset="1" stop-color="#dcebd6" stop-opacity="0.35"/></linearGradient>'
    '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
    '<marker id="arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{GRAY}"/></marker>'
    '</defs>')
rect(0, 0, W, H, "white")
rect(18, 17, 951, 430, "url(#bg)", INK, 3, rx=42)

# ================================================================= left panel
text(250, 60, "Exponent-Routed Neuron", 30, INK)

# input node
XC = 270
Y_IN = 106
circle(XC, Y_IN, 28, "url(#beige)", INK, 3)
text(XC - 36, Y_IN + 9, f'{it("x")} = ∑{sub(it("i"))} {it("w")}{sub(it("i"))} {it("a")}{sub(it("i"))}', 26, INK, "end")

# router: the sign / exponent / mantissa fields of x
RY, RH = 152, 28
cells = [("s", 22, ROUTER_FILL, ROUTER_STROKE, ROUTER_TEXT),
         ("e", 46, ROUTER_FILL, ROUTER_STROKE, ROUTER_TEXT),
         ("m", 46, MANT_FILL, MANT_STROKE, MANT_TEXT)]
rw = sum(c[1] for c in cells)
rx0 = XC - rw / 2
line(XC, Y_IN + 28, XC, RY - 2, INK, 2.6, marker="arr")
xx = rx0
for lab, cw, f, s, tc in cells:
    rect(xx, RY, cw, RH, f, s, 1.6, rx=3)
    text(xx + cw / 2, RY + RH - 8, lab, 19, tc, italic=True)
    xx += cw

# bank row
BY0, BY1 = 200, 292
rect(105, BY0, 330, BY1 - BY0, BOX_BLUE, BOX_BLUE_STROKE, 1.6, rx=16, dash="6,4")
banks = [(165, 2, False, f'bank{sub("1")}'),
         (270, 3, True, f'bank{sub(it("j"))}'),
         (375, 1, False, f'bank{sub(it("R"))}')]
BW, BH = 80, 66
YB = (BY0 + BY1) / 2
for bx, n, on, lab in banks:
    if on:
        rect(bx - BW / 2, YB - BH / 2, BW, BH, SEL_FILL, SEL_STROKE, 2.2, rx=9)
    else:
        rect(bx - BW / 2, YB - BH / 2, BW, BH, "#f7f9fc", GRAY, 1.5, rx=9, dash="5,4")
    for k in range(n):
        cx = bx + (k - (n - 1) / 2) * 19
        if on:
            circle(cx, YB - 11, 6.5, BASIS_ON)
        else:
            circle(cx, YB - 11, 6.5, "white", GRAY, 1.3)
    text(bx, YB + 23, lab, 18, INK if on else "#555", italic=True)
text(217, YB + 4, "···", 22, INK)
text(322, YB + 4, "···", 22, INK)

# routing arrows: solid to the selected bank, dashed to the alternatives
line(XC, RY + RH, XC, YB - BH / 2 - 2, INK, 2.8, marker="arr")
line(XC - 18, RY + RH, 165 + 6, YB - BH / 2 - 2, GRAY, 1.3, dash="4,4", marker="arrg")
line(XC + 18, RY + RH, 375 - 6, YB - BH / 2 - 2, GRAY, 1.3, dash="4,4", marker="arrg")

# T_j loop on the left of the row
loop_arrow(78, YB, 22)
text(40, YB + 8, f'{it("T")}{sub(it("j"))}', 24, INK)

# output node, fed by the selected bank only
Y_OUT = 336
line(XC, YB + BH / 2 + 1, XC, Y_OUT - 30, INK, 2.8, marker="arr")
circle(XC, Y_OUT, 28, "url(#green)", INK, 3)
text(XC - 40, Y_OUT + 8, "Selected bank", 22, INK, "end")

# output equation
ey = 416
text(150, ey, f'{it("f̂")}({it("x")}) =', 26, INK, "end")
sigma(178, ey, f'{it("N")}{sub(it("j"))}', f'{it("n")}=1')
text(202, ey, f'{it("w")}{subsup(it("j"), "(" + it("n") + ")")} · {it("o")}{subsup(it("j"), "(" + it("n") + ")")}({it("T")}{sub(it("j"))})', 26, INK, "start")

# ================================================================= zoom wedge
add(f'<polygon points="436,{BY0} 436,{BY1} 490,414 490,76" fill="url(#wedge)"/>')

# ================================================================ right panel
text(703, 60, f'{it("j")}-th Bank', 30, INK)
IX0, IX1, IY0, IY1 = 490, 915, 75, 415
rect(IX0, IY0, IX1 - IX0, IY1 - IY0, "url(#inner)", INK, 3, rx=42)

# row 1: x enters, lies in the bank's binade, is reduced to its mantissa residual
Y1 = 135
line(466, Y1, 514, Y1, INK, 3, marker="arr")
text(478, Y1 - 12, it("x"), 20, INK)
# binade strip
sx0, sx1 = 518, 606
t0, t1 = 530, 594
line(sx0, Y1, sx1, Y1, GRAY, 1.5)
line(t0, Y1, t1, Y1, ROUTER_STROKE, 6)
for tx in (t0, t1):
    line(tx, Y1 - 7, tx, Y1 + 7, INK, 1.6)
circle(555, Y1, 4.5, INK)
text(t0, Y1 + 24, f'2{sup(it("e"))}', 16, INK)
text(t1, Y1 + 24, f'2{sup(it("e") + "+1")}', 16, INK)
# residual box
bx0, bx1 = 632, 782
line(sx1 + 2, Y1, bx0 - 2, Y1, INK, 2.6, marker="arr")
rect(bx0, Y1 - 19, bx1 - bx0, 38, LAV, LAV_STROKE, 1.6, rx=5, dash="5,4")
text((bx0 + bx1) / 2, Y1 + 8, f'{it("ρ")} = |{it("x")}| / 2{sup(it("e"))} − 1', 22, INK)

# row 2: the bank's own bases, run for T_j steps
Y2 = 240
rect(560, Y2 - 44, 300, 88, BOX_BLUE, BOX_BLUE_STROKE, 1.6, rx=16, dash="6,4")
nj = it("N") + '<tspan baseline-shift="sub" font-size="0.75em">' + it("j") + '</tspan>'
bases = [(612, f'basis{sub(it("j") + ",1")}'), (703, f'basis{sub(it("j") + ",2")}'), (815, f'basis{sub(it("j") + "," + nj)}')]
for bx, lab in bases:
    circle(bx, Y2, 30, "#f4f7fb", INK, 2, dash="5,4")
    text(bx, Y2 + 6, lab, 16, INK, italic=True)
text(759, Y2 + 6, "···", 22, INK)
loop_arrow(535, Y2, 20)
text(535, Y2 + 48, f'{it("T")}{sub(it("j"))}', 22, INK)

# residual fans out to every basis of the bank
rc = (bx0 + bx1) / 2
for bx, _ in bases:
    line(rc, Y1 + 20, bx, Y2 - 31, INK, 2.2, marker="arr")

# row 3: weighted readout leaves the bank
Y3 = 352
import math
R_SUM = 14
for bx, _ in bases:
    dx, dy = 703 - bx, Y3 - (Y2 + 31)
    L = math.hypot(dx, dy)
    ex, ey2 = 703 - dx / L * (R_SUM + 2), Y3 - dy / L * (R_SUM + 2)
    line(bx, Y2 + 32, round(ex, 1), round(ey2, 1), INK, 2.2, marker="arr")
text(620, Y3 - 30, f'{it("w")}{subsup(it("j"), "(1)")}', 16, INK, "end")
text(712, Y3 - 44, f'{it("w")}{subsup(it("j"), "(2)")}', 16, INK, "start")
text(788, Y3 - 30, f'{it("w")}{subsup(it("j"), "(" + nj + ")")}', 16, INK, "start")
circle(703, Y3, R_SUM, "white", INK, 2.4)
line(703 - 9, Y3, 703 + 9, Y3, INK, 1.8)
line(703, Y3 - 9, 703, Y3 + 9, INK, 1.8)
line(703 + R_SUM + 1, Y3, 950, Y3, INK, 3, marker="arr")
text(880, Y3 - 12, f'{it("f̂")}({it("x")})', 20, INK)

add('</svg>')

svg = "\n".join(out)
here = os.path.dirname(os.path.abspath(__file__))
stem = os.path.join(here, "bank_neuron")
with open(stem + ".svg", "w", encoding="utf-8") as f:
    f.write(svg)
print("wrote", stem + ".svg")

if "--png" in sys.argv:
    browser = next((b for b in [
        shutil.which("chrome"), shutil.which("msedge"),
        "C:/Program Files/Google/Chrome/Application/chrome.exe",
        "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    ] if b and os.path.exists(b)), None)
    if browser is None:
        sys.exit("no Chrome/Edge found; open the .svg or render it another way")
    # (css width, device scale, suffix): 453 px wide = the reference crop's size
    for css_w, scale, suffix in [(453, 1, ".png"), (986, 1, "@2x.png"), (986, 2, "@4x.png")]:
        css_h = round(H * css_w / W)
        html = stem + ".render.html"
        with open(html, "w", encoding="utf-8") as f:
            f.write('<!doctype html><meta charset="utf-8">'
                    '<body style="margin:0;overflow:hidden;background:#fff">'
                    + svg.replace(f'width="{W}" height="{H}"', f'width="{css_w}" height="{css_h}"', 1)
                    + '</body>')
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--force-device-scale-factor={scale}", f"--window-size={css_w},{css_h}",
                        f"--screenshot={stem}{suffix}", "file:///" + html.replace(os.sep, "/")],
                       check=True, capture_output=True)
        os.remove(html)
        print("wrote", stem + suffix, f"({css_w * scale} x {css_h * scale})")
