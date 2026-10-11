"""Demo visual comparativa: la misma secuencia y los mismos frames, un renglon por metodo.
Cada vehiculo del GT se pinta VERDE si el metodo lo detecto (IoU >= 0.5) y ROJO si lo perdio (FN);
las predicciones sin pareja (FP) van en NARANJA. Arriba a la izquierda de cada recuadro: TP / FN / FP del frame.
Uso: python -I scripts/compare_demos.py --seq M0209 --methods first-5-jobs/method1_yolo11_sort first-5-jobs/method3_rtdetrm_botsort \\
         --labels "YOLOv11 + SORT" "RT-DETR + BoT-SORT" --auto 3
     (--frames f1 f2 f3 en vez de --auto; --crop x1 y1 x2 y2 para acercar una zona)
Con --auto K elige K frames repartidos donde mas difiere el numero de vehiculos perdidos entre el primer y el ultimo metodo."""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import iou_matrix, load, load_ignore

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data"); ap.add_argument("--seq", required=True)
ap.add_argument("--methods", nargs="+", required=True, help="carpetas dentro de results/")
ap.add_argument("--labels", nargs="*"); ap.add_argument("--frames", nargs="*", type=int)
ap.add_argument("--auto", type=int, default=3); ap.add_argument("--crop", nargs=4, type=int)
ap.add_argument("--width", type=int, default=560, help="ancho de cada recuadro en px")
ap.add_argument("--out")
a = ap.parse_args()
labels = a.labels or [Path(m).name for m in a.methods]
GT_DIR = Path(a.root) / "UAV-benchmark-MOTD_v1.0" / "GT"
gt = load(GT_DIR / f"{a.seq}_gt.txt"); ign = load_ignore(GT_DIR / f"{a.seq}_gt_ignore.txt")
preds = [load(Path("results") / m / f"{a.seq}.txt", ign) for m in a.methods]


def match(f, pr):
    """-> (GT detectados: set de indices, FP: lista de indices de predicciones sin pareja)"""
    g, p = gt.get(f, []), pr.get(f, [])
    if not g or not p: return set(), list(range(len(p)))
    iou = iou_matrix([x[1:] for x in g], [x[1:] for x in p]); r, c = linear_sum_assignment(-iou)
    ok = [(i, j) for i, j in zip(r, c) if iou[i, j] >= 0.5]
    return {i for i, _ in ok}, [j for j in range(len(p)) if j not in {j for _, j in ok}]

frames = a.frames
if not frames:
    common = [f for f in sorted(gt) if all(f in p for p in preds)]
    diff = np.array([len(gt[f]) - len(match(f, preds[0])[0]) - (len(gt[f]) - len(match(f, preds[-1])[0])) for f in common])
    # frames con mayor diferencia, repartidos en el tiempo (un candidato por tramo)
    chunks = np.array_split(np.arange(len(common)), a.auto)
    frames = [common[c[np.argmax(np.abs(diff[c]))]] for c in chunks if len(c)]
print("frames:", frames)

font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 15)
big = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 19)
tiles = []
for mi, pr in enumerate(preds):
    row = []
    for f in frames:
        im = Image.open(Path(a.root) / "UAV-benchmark-M" / a.seq / f"img{f:06d}.jpg").convert("RGB"); d = ImageDraw.Draw(im)
        hit, fp = match(f, pr); g, p = gt.get(f, []), pr.get(f, [])
        for i, (_, x, y, w, h) in enumerate(g):
            d.rectangle([x, y, x + w, y + h], outline=(40, 220, 70) if i in hit else (235, 30, 30), width=2 if i in hit else 4)
        for j in fp:
            _, x, y, w, h = p[j]; d.rectangle([x, y, x + w, y + h], outline=(255, 150, 0), width=2)
        d.rectangle([0, 0, 330, 30], fill=(0, 0, 0)); d.text((6, 5), f"f{f}  TP {len(hit)}  FN {len(g) - len(hit)}  FP {len(fp)}", fill=(255, 255, 255), font=font)
        if a.crop: im = im.crop(tuple(a.crop))
        s = a.width / im.width; row.append(im.resize((a.width, int(im.height * s)), Image.LANCZOS))
    tiles.append(row)

lw, gap = 190, 6
tw, th = tiles[0][0].size
W = lw + len(frames) * (tw + gap); H = len(preds) * (th + gap) + 34
sheet = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(sheet)
for r, row in enumerate(tiles):
    y = 34 + r * (th + gap)
    d.text((8, y + th // 2 - 22), labels[r], fill=(27, 58, 102), font=big)
    for c, t in enumerate(row): sheet.paste(t, (lw + c * (tw + gap), y))
d.text((lw, 6), "verde: vehiculo detectado   rojo: vehiculo perdido (FN)   naranja: falso positivo", fill=(60, 60, 60), font=font)
out = Path(a.out) if a.out else Path("presentacion/fig") / f"compare_{a.seq}.jpg"
out.parent.mkdir(parents=True, exist_ok=True); sheet.save(out, quality=92); print("guardado", out, sheet.size)
