"""Utilidades de evaluacion compartidas (lectura de GT/predicciones MOT, regiones ignoradas, IoU)."""
from pathlib import Path
import numpy as np


def iou_matrix(g, p):
    """IoU entre cajas tlwh g (N,4) y p (M,4)."""
    g, p = np.array(g, dtype=float).reshape(-1, 4), np.array(p, dtype=float).reshape(-1, 4)
    gx2, gy2, px2, py2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3], p[:, 0] + p[:, 2], p[:, 1] + p[:, 3]
    iw = np.clip(np.minimum(gx2[:, None], px2[None]) - np.maximum(g[:, 0, None], p[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(gy2[:, None], py2[None]) - np.maximum(g[:, 1, None], p[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (g[:, 2, None] * g[:, 3, None] + p[None, :, 2] * p[None, :, 3] - inter + 1e-9)


def iou_dist(g, p, thr=0.5):
    """Distancia 1-IoU; NaN (no emparejable) si IoU < thr. (motmetrics 1.4 usa np.asfarray, roto en NumPy 2)"""
    iou = iou_matrix(g, p)
    d = 1 - iou
    d[iou < thr] = np.nan
    return d


def load_ignore(path):
    """frame -> [[l,t,w,h], ...] de *_gt_ignore.txt"""
    d = {}
    if Path(path).exists():
        for l in Path(path).read_text().split():
            r = list(map(float, l.split(",")))
            d.setdefault(int(r[0]), []).append(r[2:6])
    return d


def load(path, ignore=None):
    """frame -> [(id, l, t, w, h)]. Con `ignore` replica el toolkit oficial: descarta predicciones
    totalmente dentro de una region ignorada del mismo frame."""
    d = {}
    for l in Path(path).read_text().split():
        r = l.split(",")
        f, b = int(r[0]), tuple(map(float, r[2:6]))
        if ignore and any(b[0] > i[0] and b[1] > i[1] and b[0] + b[2] < i[0] + i[2] and b[1] + b[3] < i[1] + i[3]
                          for i in ignore.get(f, [])):
            continue
        d.setdefault(f, []).append((int(r[1]), *b))
    return d
