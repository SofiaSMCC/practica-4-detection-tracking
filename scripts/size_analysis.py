"""Analisis del impacto de SAHI en vehiculos pequenos: recall de DETECCION por tamano de vehiculo (ignora los IDs).
Para cada metodo y cada frame se empareja GT con predicciones (hungaro, IoU >= 0.5); un GT sin pareja es un FN.
El tamano es sqrt(ancho*alto) en pixeles; los cortes son terciles del GT de las secuencias evaluadas (iguales para todos
los metodos). Salida: tabla en pantalla y results/size_recall.csv.
Uso: python -I scripts/size_analysis.py --names method2_yolo26_bytetrack method4_sahi_yolo26_bytetrack"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import iou_matrix, load, load_ignore

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data")
ap.add_argument("--names", nargs="+", required=True)
a = ap.parse_args()
GT_DIR = Path(a.root) / "UAV-benchmark-MOTD_v1.0" / "GT"

timings = {n: json.loads((Path("results") / n / "timing.json").read_text()) for n in a.names}
seqs = sorted(set.intersection(*[set(t) for t in timings.values()]))   # solo secuencias que tienen TODOS los metodos
frames = {s: min(t[s]["frames"] for t in timings.values()) for s in seqs}

sizes = np.concatenate([[np.sqrt(b[3] * b[4]) for fr, rows in load(GT_DIR / f"{s}_gt.txt").items() if fr <= frames[s] for b in rows] for s in seqs])
q1, q2 = np.percentile(sizes, [33.3, 66.7])
bins = [("pequeno", 0, q1), ("mediano", q1, q2), ("grande", q2, 1e9)]
print(f"{len(seqs)} secuencias, {len(sizes)} vehiculos GT. Cortes (sqrt(area), px): {q1:.1f} / {q2:.1f}")

rows = []
for n in a.names:
    hit = {b[0]: 0 for b in bins}; tot = {b[0]: 0 for b in bins}; fp = 0
    for s in seqs:
        gt = load(GT_DIR / f"{s}_gt.txt"); pr = load(Path("results") / n / f"{s}.txt", load_ignore(GT_DIR / f"{s}_gt_ignore.txt"))
        for f in range(1, frames[s] + 1):
            g, p = [x[1:] for x in gt.get(f, [])], [x[1:] for x in pr.get(f, [])]
            matched = set()
            if g and p:
                iou = iou_matrix(g, p); r, c = linear_sum_assignment(-iou)
                matched = {i for i, j in zip(r, c) if iou[i, j] >= 0.5}
                fp += len(p) - len(matched)
            else:
                fp += len(p)
            for i, b in enumerate(g):
                sz = np.sqrt(b[2] * b[3])
                name = next(k for k, lo, hi in bins if lo <= sz < hi)
                tot[name] += 1; hit[name] += i in matched
    rows.append({"metodo": n, **{f"recall_{k}": hit[k] / max(tot[k], 1) for k, _, _ in bins}, "FP": fp, **{f"n_{k}": tot[k] for k, _, _ in bins}})
df = pd.DataFrame(rows).set_index("metodo")
print(df.round(3).to_string())
df.to_csv("results/size_recall.csv")
