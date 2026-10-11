"""Fusiona las detecciones cacheadas de varios detectores (tipo Weighted Boxes Fusion) -> results/_cache/<tag>/<seq>.pkl
Para cada frame: se agrupan las cajas de distintos modelos con IoU >= thr (voraz, de mayor a menor confianza); la caja fusionada
es el promedio ponderado por confianza y su score = (suma de confianzas del grupo) / n_modelos. Una caja que ve un solo modelo
queda con la confianza dividida por n_modelos, asi que hace falta consenso para pasar umbrales altos.
Uso: python -I scripts/fuse_cache.py --inputs yolo26 rtdetr --tag fuse [--thr 0.55]"""
import argparse, pickle, sys, os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import find_sequences

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True); ap.add_argument("--tag", default="fuse")
ap.add_argument("--cache", default="results/_cache"); ap.add_argument("--root", default="data")
ap.add_argument("--thr", type=float, default=0.55); ap.add_argument("--min_conf", type=float, default=0.05)
a = ap.parse_args()
VISDRONE_VEH = {3, 4, 5, 8}      # car, van, truck, bus en RT-DETR (VisDrone); los de YOLO ya son solo vehiculos


def iou_one(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (box[2] - box[0]) * (box[3] - box[1]); a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def fuse(frames_per_model):
    """frames_per_model: lista (un elemento por modelo) de arrays (N,5) [x1,y1,x2,y2,conf]. -> (M,6) [..., conf, 0]"""
    n = len(frames_per_model)
    pool = [(m, row) for m, f in enumerate(frames_per_model) for row in f]
    pool.sort(key=lambda t: -t[1][4])
    centers, groups = [], []          # caja fusionada actual y miembros de cada grupo
    for m, row in pool:
        if centers:
            iou = iou_one(row, np.array(centers))
            for gi in np.argsort(-iou):
                if iou[gi] < a.thr: break
                if all(mm != m for mm, _ in groups[gi]):       # un solo miembro por modelo
                    groups[gi].append((m, row))
                    w = np.array([r[4] for _, r in groups[gi]]); b = np.array([r[:4] for _, r in groups[gi]])
                    centers[gi] = (b * w[:, None]).sum(0) / w.sum()
                    break
            else:
                centers.append(row[:4].copy()); groups.append([(m, row)]); continue
            if iou[gi] >= a.thr and groups[gi][-1][1] is row: continue
            centers.append(row[:4].copy()); groups.append([(m, row)])
        else:
            centers.append(row[:4].copy()); groups.append([(m, row)])
    out = [[*centers[i], sum(r[4] for _, r in g) / n, 0] for i, g in enumerate(groups)]
    return np.array(out, dtype=np.float32).reshape(-1, 6)


out_dir = Path(a.cache) / a.tag; out_dir.mkdir(parents=True, exist_ok=True)
for seq, _ in find_sequences(a.root):
    f_out = out_dir / f"{seq}.pkl"
    if f_out.exists(): continue
    src = [pickle.load(open(Path(a.cache) / n / f"{seq}.pkl", "rb")) for n in a.inputs]
    nfr = min(len(s["dets"]) for s in src); res = []
    for k in range(nfr):
        per = []
        for name, s in zip(a.inputs, src):
            d = s["dets"][k]
            if len(d):
                keep = d[:, 4] >= a.min_conf
                if name == "rtdetr": keep &= np.isin(d[:, 5].astype(int), list(VISDRONE_VEH))
                d = d[keep][:, :5]
            per.append(d if len(d) else np.zeros((0, 5)))
        res.append(fuse(per))
    tmp = f_out.with_suffix(".tmp"); pickle.dump({"dets": res, "t_det": sum(s["t_det"] for s in src), "frames": nfr}, open(tmp, "wb"))
    os.replace(tmp, f_out); print("fusionado", seq, nfr, "frames", flush=True)
