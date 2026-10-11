"""Genera las figuras reales de la presentacion (frames del dataset con GT, collage, slices de SAHI) y stats.json.
Uso: python -I presentacion/make_figures.py   (desde la raiz del proyecto)"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

D = Path("data"); GT = D / "UAV-benchmark-MOTD_v1.0" / "GT"; OUT = Path("presentacion/fig"); OUT.mkdir(parents=True, exist_ok=True)
seqs = sorted(p.name.replace(" ", "").replace("_attr.txt", "") for p in (D / "M_attr" / "test").glob("*_attr.txt"))

def frame(seq, n): return Image.open(D / "UAV-benchmark-M" / seq / f"img{n:06d}.jpg").convert("RGB")
def rows(path, n):
    return [tuple(map(float, l.split(",")[2:6])) for l in path.read_text().split() if int(l.split(",")[0]) == n] if path.exists() else []
def font(sz):
    for f in ("/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"):
        if Path(f).exists(): return ImageFont.truetype(f, sz)
    return ImageFont.load_default()

# 1) estadisticas por secuencia (para los graficos nativos)
stats = []
for s in seqs:
    n = len(list((D / "UAV-benchmark-M" / s).glob("img*.jpg")))
    b = sum(1 for _ in (GT / f"{s}_gt.txt").read_text().split())
    stats.append(dict(seq=s, frames=n, boxes=b, per_frame=round(b / n, 1)))
json.dump(stats, open("presentacion/stats.json", "w"), indent=1)

# 2) collage 2x2 de secuencias distintas (como el collage de la plantilla)
pick = [("M0203", 300), ("M0403", 200), ("M0701", 400), ("M1301", 300)]
w, h = 512, 270
col = Image.new("RGB", (w * 2 + 6, h * 2 + 6), "white")
for k, (s, n) in enumerate(pick):
    col.paste(frame(s, n).resize((w, h), Image.LANCZOS), ((k % 2) * (w + 6), (k // 2) * (h + 6)))
col.save(OUT / "collage.jpg", quality=92)

# 3) frame con GT (verde) y regiones ignoradas (rojo)
s, n = "M0403", 200
im = frame(s, n); dr = ImageDraw.Draw(im, "RGBA")
for (x, y, bw, bh) in rows(GT / f"{s}_gt_ignore.txt", n): dr.rectangle([x, y, x + bw, y + bh], fill=(200, 30, 30, 70), outline=(200, 30, 30, 255), width=3)
for (x, y, bw, bh) in rows(GT / f"{s}_gt.txt", n): dr.rectangle([x, y, x + bw, y + bh], outline=(40, 200, 70, 255), width=2)
im.save(OUT / "gt_frame.jpg", quality=92)

# 4) slices de SAHI (config usada: slice 640, solape 20%) sobre un frame real
im = frame("M0701", 400); W, H = im.size; S, ov = 640, 0.2; step = int(S * (1 - ov))
xs = sorted({min(x, W - S) for x in range(0, W, step)}); print("slices en x:", xs, "(alto", H, "< 640: una sola fila)")
ov_l = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov_l)
cols = [(30, 90, 200), (255, 150, 20)]
for i, x0 in enumerate(xs):
    c = cols[i % 2]; d.rectangle([x0, 0, x0 + S, H - 1], fill=c + (45,), outline=c + (255,), width=5)
    d.text((x0 + 14, 10 + 36 * i), f"slice {i + 1}", fill=c + (255,), font=font(30))
Image.alpha_composite(im.convert("RGBA"), ov_l).convert("RGB").save(OUT / "sahi_slices.jpg", quality=92)
print("ok", len(stats), "secuencias")

# 5) GMC: flujo optico disperso + estimacion afin (RANSAC) entre dos frames de una secuencia con movimiento de camara
import cv2, numpy as np
seq, f0, dt = "M0801", 50, 5
a, b = frame(seq, f0), frame(seq, f0 + dt)
ga, gb = cv2.cvtColor(np.array(a), cv2.COLOR_RGB2GRAY), cv2.cvtColor(np.array(b), cv2.COLOR_RGB2GRAY)
p0 = cv2.goodFeaturesToTrack(ga, maxCorners=300, qualityLevel=0.01, minDistance=10)
p1, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None); ok = st[:, 0] == 1
q0, q1 = p0[ok][:, 0], p1[ok][:, 0]
M, inl = cv2.estimateAffinePartial2D(q0, q1, cv2.RANSAC); inl = inl[:, 0] == 1
im = b.copy(); dr = ImageDraw.Draw(im)
for (x0, y0), (x1, y1), g in zip(q0, q1, inl):
    c = (40, 220, 80) if g else (230, 40, 40)
    dr.line([x0, y0, x1, y1], fill=c, width=2); dr.ellipse([x1 - 3, y1 - 3, x1 + 3, y1 + 3], fill=c)
im.save(OUT / "gmc_flow.jpg", quality=92)
tx, ty = float(M[0, 2]), float(M[1, 2]); ang = float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))); sc = float(np.hypot(M[0, 0], M[1, 0]))
json.dump(dict(seq=seq, f0=f0, dt=dt, pts=int(ok.sum()), inliers=int(inl.sum()), tx=round(tx, 1), ty=round(ty, 1), rot_deg=round(ang, 2), scale=round(sc, 3)),
          open("presentacion/gmc_stats.json", "w"))
print("GMC", seq, f0, f0 + dt, "t=(%.1f, %.1f) px rot=%.2f deg scale=%.3f inliers %d/%d" % (tx, ty, ang, sc, inl.sum(), ok.sum()))
