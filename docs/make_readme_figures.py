"""Draw the README figures in docs/img/ from the model's real data.

Usage: python docs/make_readme_figures.py
Needs: numpy. Reads model/weights_ternary.npz and model/test_vectors.txt.
"""
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
out = root / "docs" / "img"
W = np.load(root / "model" / "weights_ternary.npz")["arr_0"]  # (10, 64)
rows = [l.split() for l in (root / "model" / "test_vectors.txt").read_text().splitlines() if not l.startswith("#")]
LV = np.array([[int(c) for c in r[0]] for r in rows])
LAB = np.array([int(r[1]) for r in rows])
PRED = np.array([int(r[2]) for r in rows])
CNT = np.array([[int(x) for x in r[3:13]] for r in rows])

SANS = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
BG, GRID, INK, INK2, MUTED = "#0d1117", "#30363d", "#e6edf3", "#c9d1d9", "#8b949e"
BLUE, ORANGE, AQUA, VIOLET = "#3987e5", "#d95926", "#199e70", "#9085e9"
NEUTRAL = "#21262d"


def t(x, y, s, size=12.5, fill=INK, anchor="start", weight=400, font=SANS, extra=""):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="{font}"{extra}>{s}</text>')


def svg(w, h, title, sub, body):
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">',
            f"<title>{title}</title>", f'<rect width="{w}" height="{h}" rx="12" fill="{BG}"/>',
            t(30, 36, title, 16, weight=600), t(30, 57, sub, 12.5, MUTED)]
    return "\n".join(head + body + ["</svg>"])


def save(name, s):
    (out / name).write_text(s, encoding="utf-8")
    print("wrote", name)


def rate(lv, T=16):
    tt = np.arange(T + 1)[:, None]
    acc = (tt * lv[None, :]) // 4
    return (acc[1:] > acc[:-1]).astype(int)


def run(lv):
    """Integer model; returns membrane after reset per step (T,10) and spikes (T,10)."""
    s = rate(lv)
    v = np.zeros(10, dtype=int)
    vs, zs = [], []
    for st in s:
        v = v - (v >> 4)
        v = np.clip(v + W @ st, -128, 127)
        pre = v.copy()
        z = (v >= 8).astype(int)
        v = v - 8 * z
        vs.append(pre)
        zs.append(z)
    return np.array(vs), np.array(zs)


# ---------------------------------------------------------------- 1. weights
def weights():
    c, gap, x0, y0 = 13, 30, 30, 96
    body = []
    for n in range(10):
        gx = x0 + n * (8 * c + gap) - (0 if n < 10 else 0)
        body.append(t(gx + 4 * c, y0 - 10, f"digit {n}", 13, INK2, "middle", 600))
        for p in range(64):
            r, k = divmod(p, 8)
            col = {1: BLUE, -1: ORANGE, 0: NEUTRAL}[int(W[n, p])]
            body.append(f'<rect x="{gx + k * c + 1}" y="{y0 + r * c + 1}" width="{c - 2}" height="{c - 2}" rx="2" fill="{col}"/>')
    ly = y0 + 8 * c + 34
    for i, (col, lab) in enumerate([(BLUE, "+1 excites"), (ORANGE, "−1 inhibits"), (NEUTRAL, "0 no connection")]):
        body.append(f'<rect x="{30 + i * 170}" y="{ly - 11}" width="12" height="12" rx="2" fill="{col}" stroke="{GRID}"/>')
        body.append(t(48 + i * 170, ly, lab, 12.5, INK2))
    body.append(t(1350 - 30, ly, f"{int((W != 0).sum())} of 640 weights non-zero", 12, MUTED, "end", font=MONO))
    return svg(1350, ly + 26, "The 640 hardwired weights, one 8×8 map per output neuron",
               "Each neuron learned a template of its digit: blue pixels push it toward firing, orange pixels hold it back.", body)


# ---------------------------------------------------------------- 2. accuracy by digit
def accuracy():
    acc = [100 * (PRED[LAB == d] == d).mean() for d in range(10)]
    n = [(LAB == d).sum() for d in range(10)]
    x0, base, h, bw, step = 64, 330, 240, 46, 82
    body = []
    for g in (0, 25, 50, 75, 100):
        y = base - h * g / 100
        body.append(f'<line x1="{x0 - 10}" y1="{y}" x2="{x0 + 10 * step}" y2="{y}" stroke="{GRID}"/>')
        body.append(t(x0 - 16, y + 4, f"{g}%", 11, MUTED, "end"))
    overall = 100 * (PRED == LAB).mean()
    yo = base - h * overall / 100
    body.append(f'<line x1="{x0 - 10}" y1="{yo:.1f}" x2="{x0 + 10 * step}" y2="{yo:.1f}" stroke="{MUTED}" stroke-dasharray="5 4"/>')
    body.append(t(x0 + 10 * step + 6, yo + 4, f"overall {overall:.2f}%", 11.5, INK2))
    for d in range(10):
        x = x0 + d * step + (step - bw) / 2
        bh = h * acc[d] / 100
        body.append(f'<path d="M{x} {base}v-{bh - 4:.1f}a4 4 0 0 1 4 -4h{bw - 8}a4 4 0 0 1 4 4v{bh - 4:.1f}z" fill="{BLUE}"/>')
        body.append(t(x + bw / 2, base - bh - 8, f"{acc[d]:.1f}%", 12, INK, "middle", font=MONO))
        body.append(t(x + bw / 2, base + 22, str(d), 15, INK, "middle", 600))
        body.append(t(x + bw / 2, base + 40, f"n={n[d]}", 10.5, MUTED, "middle"))
    return svg(1000, 400, "Accuracy by digit, all 10,000 MNIST test images",
               "Share of each digit's test images the chip classifies correctly (bit-exact chip model).", body)


# ---------------------------------------------------------------- 3. confusion matrix
def confusion():
    m = np.zeros((10, 10), int)
    for a, b in zip(LAB, PRED):
        m[a, b] += 1
    rown = m / m.sum(1, keepdims=True)
    c, x0, y0 = 46, 130, 110
    ramp = ["#0d1f3a", "#12305a", "#17427c", "#1c569e", "#2a6cc0", "#3987e5", "#6aa6ee", "#a5c9f5"]
    body = [t(x0 + 5 * c, y0 - 30, "predicted digit", 12.5, INK2, "middle")]
    body.append(t(x0 - 62, y0 + 5 * c, "true digit", 12.5, INK2, "middle", extra=f' transform="rotate(-90 {x0 - 62} {y0 + 5 * c})"'))
    for k in range(10):
        body.append(t(x0 + k * c + c / 2, y0 - 8, str(k), 13, INK, "middle", 600))
        body.append(t(x0 - 12, y0 + k * c + c / 2 + 5, str(k), 13, INK, "end", 600))
    for i in range(10):
        for j in range(10):
            f = rown[i, j]
            if i == j:
                col = ramp[-1] if f > 0.85 else ramp[-2]
            else:
                col = ramp[min(len(ramp) - 3, int(f / 0.15 * (len(ramp) - 3)))] if m[i, j] else "#111821"
            body.append(f'<rect x="{x0 + j * c + 1}" y="{y0 + i * c + 1}" width="{c - 2}" height="{c - 2}" rx="3" fill="{col}"/>')
            if m[i, j]:
                ink = "#0d1117" if i == j else (INK if f > 0.04 else MUTED)
                body.append(t(x0 + j * c + c / 2, y0 + i * c + c / 2 + 4, str(m[i, j]), 11 if m[i, j] < 1000 else 10,
                              ink, "middle", 600 if i == j else 400, MONO))
    # biggest confusions list
    off = sorted(((m[i, j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:6]
    lx = x0 + 10 * c + 50
    body.append(t(lx, y0 + 6, "Most common mistakes", 14, INK, weight=600))
    for k, (v, i, j) in enumerate(off):
        y = y0 + 40 + k * 34
        wbar = 160 * v / off[0][0]
        body.append(t(lx, y, f"{i} → {j}", 13, INK, weight=600))
        body.append(f'<rect x="{lx + 56}" y="{y - 12}" width="{wbar:.1f}" height="16" rx="3" fill="{ORANGE}"/>')
        body.append(t(lx + 64 + wbar, y + 1, f"{v} images", 12, MUTED, font=MONO))
    body.append(t(lx, y0 + 40 + 6 * 34 + 10, "Downsampled to 8×8 at 2 bits, a 5 and a 3", 12, MUTED))
    body.append(t(lx, y0 + 40 + 6 * 34 + 28, "or a 4 and a 9 differ by only a few pixels.", 12, MUTED))
    return svg(1000, y0 + 10 * c + 30, "Confusion matrix, 10,000 test images",
               "Rows: true digit. Columns: the chip's answer. The bright diagonal is correct; everything else is a mistake.", body)


# ---------------------------------------------------------------- 4. membrane traces
def membrane():
    vs, zs = run(LV[0])
    series = [(7, BLUE, "neuron 7 (winner)"), (9, AQUA, "neuron 9 (runner-up)"), (6, ORANGE, "neuron 6 (suppressed)")]
    x0, x1, y0, y1 = 80, 860, 90, 370
    sel = vs[:, [n for n, _, _ in series]]
    vmin, vmax = int(np.floor(sel.min() / 2) * 2) - 2, int(np.ceil(sel.max() / 2) * 2) + 2
    X = lambda s: x0 + (x1 - x0) * s / 15
    Y = lambda v: y1 - (y1 - y0) * (v - vmin) / (vmax - vmin)
    body = []
    for g in range(vmin, vmax + 1, 2):
        body.append(f'<line x1="{x0}" y1="{Y(g):.1f}" x2="{x1}" y2="{Y(g):.1f}" stroke="{GRID}" stroke-width="{1.4 if g == 0 else 0.8}"/>')
        body.append(t(x0 - 10, Y(g) + 4, str(g), 11, MUTED, "end"))
    body.append(f'<line x1="{x0}" y1="{Y(8):.1f}" x2="{x1}" y2="{Y(8):.1f}" stroke="#e66767" stroke-width="1.6" stroke-dasharray="6 4"/>')
    body.append(t(x1 + 8, Y(8) + 4, "threshold θ = 8", 12, INK2))
    for s in range(16):
        body.append(t(X(s), y1 + 20, str(s), 11, MUTED, "middle"))
    body.append(t((x0 + x1) / 2, y1 + 40, "timestep", 12, INK2, "middle"))
    body.append(t(x0 - 50, (y0 + y1) / 2, "membrane v", 12, INK2, "middle", extra=f' transform="rotate(-90 {x0 - 50} {(y0 + y1) / 2})"'))
    for n, col, lab in series:
        pts = " ".join(f"{X(s):.1f},{Y(min(vmax, max(vmin, vs[s, n]))):.1f}" for s in range(16))
        body.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2" stroke-linejoin="round"/>')
        for s in range(16):
            v = min(vmax, max(vmin, vs[s, n]))
            if zs[s, n]:
                body.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="6" fill="{col}" stroke="{BG}" stroke-width="2"/>')
                body.append(f'<path d="M{X(s):.1f} {Y(v) - 9:.1f}v-12" stroke="{col}" stroke-width="2"/>')
            else:
                body.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3" fill="{col}"/>')
        body.append(t(x1 + 8, Y(min(vmax, max(vmin, vs[15, n]))) + 4 + (14 if n == 6 else 0), f"{lab}: {zs[:, n].sum()} spikes", 12, INK2))
    for i, (n, col, lab) in enumerate(series):
        body.append(f'<rect x="{30 + i * 220}" y="{y1 + 62}" width="14" height="4" rx="2" fill="{col}"/>')
        body.append(t(50 + i * 220, y1 + 68, lab, 12, INK2))
    body.append(t(30 + 3 * 220, y1 + 68, "large dot = spike, then v −= 8", 12, MUTED))
    return svg(1100, y1 + 90, "Inside the chip: membrane voltages while it reads a 7",
               "Values before reset at each timestep, from the bit-exact integer model (test image 0).", body)


# ---------------------------------------------------------------- 5. gallery
def gallery():
    n, c, per = 30, 7, 10
    grey = ["#0d1117", "#3b4552", "#8b949e", "#f0f6fc"]
    body = []
    for i in range(n):
        r, k = divmod(i, per)
        gx, gy = 30 + k * 94, 84 + r * 112
        ok = PRED[i] == LAB[i]
        body.append(f'<rect x="{gx - 3}" y="{gy - 3}" width="{8 * c + 6}" height="{8 * c + 6}" rx="4" fill="none" stroke="{AQUA if ok else ORANGE}" stroke-width="2"/>')
        for p in range(64):
            rr, kk = divmod(p, 8)
            body.append(f'<rect x="{gx + kk * c}" y="{gy + rr * c}" width="{c}" height="{c}" fill="{grey[LV[i, p]]}"/>')
        body.append(t(gx + 4 * c, gy + 8 * c + 20, f"chip: {PRED[i]}", 12.5, INK, "middle", 600))
        body.append(t(gx + 4 * c, gy + 8 * c + 36, ("✓" if ok else f"✗ was {LAB[i]}"), 11.5, AQUA if ok else ORANGE, "middle"))
    nok = int((PRED[:n] == LAB[:n]).sum())
    return svg(960, 84 + 3 * 112 + 6, f"What the chip sees: the first {n} test images at 8×8, 2 bits",
               f"Exactly the input the host sends. Green: classified correctly ({nok}/{n}). Orange: mistakes, with the true digit.", body)


# ---------------------------------------------------------------- 6. design sweep
def sweep():
    data = [("8×8 → 10", 79.3, "1x1  ✓ chip", True), ("8×8 → 16 → 10", 82.4, "too big for 1x1", False),
            ("8×8 → 24 → 10", 86.8, "1x2 tiles", False), ("10×10 → 10", 87.9, "1x1, too full", False),
            ("8×8 → 32 → 10", 87.3, "1x2+ tiles", False)]
    x0, body = 250, []
    scale = 520 / 100
    for g in (0, 20, 40, 60, 80, 100):
        x = x0 + g * scale
        body.append(f'<line x1="{x}" y1="78" x2="{x}" y2="{84 + 36 * len(data)}" stroke="{GRID}"/>')
        body.append(t(x, 72, f"{g}%", 11, MUTED, "middle"))
    for i, (name, a, note, chosen) in enumerate(data):
        y = 90 + 36 * i
        w = a * scale
        body.append(t(x0 - 14, y + 15, name, 13, INK, "end", 600 if chosen else 400))
        body.append(f'<path d="M{x0} {y}h{w - 4:.1f}a4 4 0 0 1 4 4v14a4 4 0 0 1 -4 4h-{w - 4:.1f}z" fill="{BLUE if chosen else "#2d4a6e"}"/>')
        body.append(t(x0 + w + 10, y + 15, f"{a:.1f}%  ·  {note}", 12, INK if chosen else MUTED, font=MONO))
    return svg(1000, 110 + 36 * len(data), "Accuracy vs. size: why the chip is a single layer",
               "MNIST test accuracy of each network the training sweep tried, and where it would fit.", body)


# ---------------------------------------------------------------- 7. protocol waveform
def protocol():
    x0, x1 = 150, 970
    sig = [("clk", "clk"), ("ui[7:0]", "bus"), ("STROBE", [(0, 0), (40, 1), (120, 0), (520, 1), (600, 0)]),
           ("BUSY", [(0, 0), (75, 1), (440, 0), (555, 1), (900, 0)]), ("DONE", [(0, 0)]),
           ("STEP_START", [(0, 1), (75, 0)])]
    body, y = [], 100
    X = lambda u: x0 + (x1 - x0) * u / 900
    for name, kind in sig:
        body.append(t(x0 - 16, y + 14, name, 13, INK, "end", 600, MONO))
        if kind == "clk":
            d = f"M{X(0)} {y + 20}"
            for k in range(0, 900, 18):
                d += f"V{y + 2}H{X(k + 9):.1f}V{y + 20}H{X(k + 18):.1f}"
            body.append(f'<path d="{d}" fill="none" stroke="{MUTED}" stroke-width="1"/>')
        elif kind == "bus":
            for a, b, lab in [(0, 20, ""), (20, 480, "byte b"), (480, 900, "byte b+1")]:
                body.append(f'<path d="M{X(a) + 4:.1f} {y + 2}H{X(b) - 4:.1f}L{X(b):.1f} {y + 11}L{X(b) - 4:.1f} {y + 20}H{X(a) + 4:.1f}L{X(a):.1f} {y + 11}z" fill="#16263b" stroke="{BLUE}" stroke-width="1.4"/>')
                if lab:
                    body.append(t((X(a) + X(b)) / 2, y + 15, lab, 12, INK, "middle", font=MONO))
        else:
            pts, lvl = [], kind[0][1]
            for k, (u, v) in enumerate(kind):
                yy = y + (2 if v else 20)
                if k:
                    pts.append(f"{X(u):.1f},{y + (2 if lvl else 20)}")
                pts.append(f"{X(u):.1f},{yy}")
                lvl = v
            pts.append(f"{X(900):.1f},{y + (2 if lvl else 20)}")
            col = {"STROBE": ORANGE, "BUSY": AQUA, "DONE": VIOLET, "STEP_START": BLUE}[name]
            body.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        y += 46
    for u, lab in [(40, "host raises STROBE"), (75, "chip latches byte, BUSY = 1"), (440, "80-clock pass done"), (520, "next byte")]:
        body.append(f'<line x1="{X(u):.1f}" y1="92" x2="{X(u):.1f}" y2="{y - 20}" stroke="{GRID}" stroke-dasharray="3 3"/>')
    body.append(t(X(40), y + 4, "① host puts byte on ui, raises STROBE ≥ 4 clocks", 12, INK2))
    body.append(t(X(40), y + 24, "② chip syncs the edge, latches the byte, BUSY = 1 for ~90 clocks (8 pixels × 10 neurons)", 12, INK2))
    body.append(t(X(40), y + 44, "③ BUSY = 0: send the next byte. After 16 × 8 bytes, DONE = 1 and uo[3:0] holds the digit", 12, INK2))
    return svg(1000, y + 64, "Talking to the chip: the byte handshake",
               "Two bytes of one timestep. All handshake inputs are synchronised, so any microcontroller can drive it.", body)


save("weights.svg", weights())
save("accuracy.svg", accuracy())
save("confusion.svg", confusion())
save("membrane.svg", membrane())
save("gallery.svg", gallery())
save("sweep.svg", sweep())
save("protocol.svg", protocol())
