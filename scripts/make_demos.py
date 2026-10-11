"""Demos visuales de tracking (casos exitosos / de fallo): dibuja cajas con ID y estela sobre varios frames de una secuencia.
  Dibujar:   python -I scripts/make_demos.py --method method3_rtdetrm_botsort --seq M0403 --frames 100 130 160 [--crop x1 y1 x2 y2] [--gt]
  Buscar:    python -I scripts/make_demos.py --method method3_rtdetrm_botsort --seq M0403 --find
             -> lista cambios de ID (candidatos a "fallo") y los vehiculos con el track mas largo sin cambiar de ID ("exito").
Salida: presentacion/fig/demo_<metodo>_<seq>.jpg"""
import argparse, colorsys, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import iou_matrix, load, load_ignore

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data"); ap.add_argument("--method", required=True); ap.add_argument("--seq", required=True)
ap.add_argument("--frames", nargs="*", type=int); ap.add_argument("--crop", nargs=4, type=int)
ap.add_argument("--trail", type=int, default=20, help="frames de estela"); ap.add_argument("--gt", action="store_true", help="dibujar tambien el GT (blanco)")
ap.add_argument("--find", action="store_true")
a = ap.parse_args()
GT_DIR = Path(a.root) / "UAV-benchmark-MOTD_v1.0" / "GT"
pred = load(Path("results") / a.method / f"{a.seq}.txt", load_ignore(GT_DIR / f"{a.seq}_gt_ignore.txt"))
gt = load(GT_DIR / f"{a.seq}_gt.txt")

if a.find:
    last, run, runs, switches = {}, defaultdict(int), defaultdict(int), []
    for f in sorted(gt):
        g, p = gt[f], pred.get(f, [])
        if not g or not p: continue
        iou = iou_matrix([x[1:] for x in g], [x[1:] for x in p]); r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] < 0.5: continue
            gid, pid = g[i][0], p[j][0]
            if gid in last and last[gid] != pid: switches.append((f, gid, last[gid], pid)); run[gid] = 0
            last[gid] = pid; run[gid] += 1; runs[gid] = max(runs[gid], run[gid])
    print(f"Cambios de ID (frame, GT id, id anterior -> nuevo): {len(switches)}")
    for s in switches[:15]: print("  frame %d  GT %d: %d -> %d" % s)
    best = sorted(runs.items(), key=lambda kv: -kv[1])[:5]
    print("Vehiculos con el track mas largo sin cambiar de ID (GT id, frames seguidos):", best)
    sys.exit()

def color(i):
    h = (i * 0.61803398875) % 1.0; r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0); return int(r * 255), int(g * 255), int(b * 255)
font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 15)
tiles = []
for f in a.frames:
    im = Image.open(Path(a.root) / "UAV-benchmark-M" / a.seq / f"img{f:06d}.jpg").convert("RGB"); d = ImageDraw.Draw(im)
    if a.gt:
        for _, x, y, w, h in gt.get(f, []): d.rectangle([x, y, x + w, y + h], outline=(255, 255, 255), width=1)
    for tid, x, y, w, h in pred.get(f, []):
        c = color(tid)
        pts = []
        for k in range(max(1, f - a.trail), f + 1):   # estela: centros de este ID en los ultimos frames
            for t2, x2, y2, w2, h2 in pred.get(k, []):
                if t2 == tid: pts.append((x2 + w2 / 2, y2 + h2 / 2))
        if len(pts) > 1: d.line(pts, fill=c, width=2)
        d.rectangle([x, y, x + w, y + h], outline=c, width=2)
        d.text((x, max(0, y - 17)), str(tid), fill=c, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((8, 6), f"frame {f}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    if a.crop: im = im.crop(tuple(a.crop))
    tiles.append(im)
w, h = tiles[0].size
sheet = Image.new("RGB", (w * len(tiles) + 6 * (len(tiles) - 1), h), "white")
for i, t in enumerate(tiles): sheet.paste(t, (i * (w + 6), 0))
out = Path("presentacion/fig") / f"demo_{a.method}_{a.seq}.jpg"; out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out, quality=92); print("guardado", out, sheet.size)
