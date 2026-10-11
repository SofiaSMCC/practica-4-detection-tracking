"""Ajuste de hiperparametros del metodo 3 (RT-DETRv4 + BoT-SORT) sin volver a detectar en cada prueba.
 Etapa cache: corre RT-DETR una vez (conf >= 0.05) y guarda detecciones + movimiento de camara (GMC) por frame.
 Etapa sweep: barrido por coordenadas (un grupo de parametros a la vez, partiendo de los valores por defecto) que
              re-ejecuta SOLO el tracker sobre las detecciones cacheadas y mide MOTA/IDF1/IDS con motmetrics.
Criterio: MOTA promedio sobre las secuencias de ajuste. Se ajusta en pocas secuencias para evaluar luego en las demas.
Uso: python -I scripts/tune_method3.py --stage cache --device cpu --threads 8 --max_frames 300
     python -I scripts/tune_method3.py --stage sweep"""
import argparse, json, pickle, sys, time
from pathlib import Path
import cv2, numpy as np
import pandas as pd
import motmetrics as mm

sys.path.insert(0, str(Path(__file__).parent))
from common import Dets, find_sequences, tracker_args
from eval_utils import iou_dist, load, load_ignore

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="both", choices=["cache", "sweep", "both"])
ap.add_argument("--root", default="data")
ap.add_argument("--seqs", nargs="*", default=["M0203", "M0205", "M0208", "M0209", "M0403"])
ap.add_argument("--max_frames", type=int, default=300)
ap.add_argument("--size", default="m")
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--cache", default="results/_cache_m3")
a = ap.parse_args()
CACHE = Path(a.cache); CACHE.mkdir(parents=True, exist_ok=True)
GT_DIR = Path(a.root) / "UAV-benchmark-MOTD_v1.0" / "GT"
seqs = find_sequences(a.root, a.seqs)

# ------------------------------------------------------------------------------------------ etapa 1: cache
if a.stage in ("cache", "both"):
    import torch
    torch.set_num_threads(a.threads)
    from rtdetr_det import load_detector
    from ultralytics.trackers.utils.gmc import GMC
    detect = load_detector(a.size, a.device)
    for seq, d in seqs:
        out = CACHE / f"{seq}.pkl"
        if out.exists(): print("cache existe:", seq); continue
        gmc = GMC(method="sparseOptFlow"); dets, warps, t0 = [], [], time.time()
        for k, p in enumerate(sorted(d.glob("img*.jpg"))[: a.max_frames]):
            img = cv2.imread(str(p))
            box, scr, lab = detect(img, 0.05)
            dets.append(np.column_stack([box, scr, lab]) if len(box) else np.zeros((0, 6)))
            warps.append(np.array(gmc.apply(img, None), dtype=np.float64)[:2])    # 2x3 respecto al frame anterior
            if k % 50 == 0: print(f"{seq} frame {k} ({time.time() - t0:.0f}s)", flush=True)
        pickle.dump(dict(dets=dets, warps=warps), open(out, "wb")); print("guardado", out, flush=True)
    if a.stage == "cache": sys.exit()

# ------------------------------------------------------------------------------------------ etapa 2: sweep
from ultralytics.trackers.bot_sort import BOTSORT

class CachedGMC:
    """Sustituye a GMC: devuelve la transformacion afin ya calculada de cada frame (o la identidad si off=True)."""
    method = "cached"
    def __init__(self, warps, off=False): self.w, self.i, self.off = warps, 0, off
    def apply(self, img, dets=None):
        w = np.eye(2, 3) if self.off else self.w[self.i]; self.i += 1; return w
    def reset_params(self): self.i = 0

data = {s: pickle.load(open(CACHE / f"{s}.pkl", "rb")) for s, _ in seqs}
gts = {s: load(GT_DIR / f"{s}_gt.txt") for s, _ in seqs}
igs = {s: load_ignore(GT_DIR / f"{s}_gt_ignore.txt") for s, _ in seqs}
DEFAULT = dict(low=0.1, high=0.25, new=0.25, buffer=30, match=0.8, min_size=0, truck_conf=0.0, van=True, gmc=True)
MAP = {3: 1, 4: 1, 5: 2, 8: 3}
DUMMY = np.zeros((2, 2, 3), np.uint8)


def run(cfg):
    ta = tracker_args("botsort.yaml", track_high_thresh=cfg["high"], track_low_thresh=cfg["low"], new_track_thresh=cfg["new"],
                      track_buffer=cfg["buffer"], match_thresh=cfg["match"])
    accs = []
    for seq, _ in seqs:
        tr = BOTSORT(ta); tr.gmc = CachedGMC(data[seq]["warps"], off=not cfg["gmc"])
        pred = {}
        for f, dd in enumerate(data[seq]["dets"], 1):
            if len(dd):
                lab = dd[:, 5].astype(int); keep = np.isin(lab, [3, 5, 8] + ([4] if cfg["van"] else []))
                cls = np.array([MAP.get(l, 0) for l in lab])
                size = np.sqrt(np.clip(dd[:, 2] - dd[:, 0], 0, None) * np.clip(dd[:, 3] - dd[:, 1], 0, None))
                keep &= (dd[:, 4] >= cfg["low"]) & (size >= cfg["min_size"]) & ~((cls == 2) & (dd[:, 4] < cfg["truck_conf"]))
                dd, cls = dd[keep], cls[keep]
                D = Dets(dd[:, :4], dd[:, 4], cls)
            else:
                D = Dets(np.zeros((0, 4)), [], [])
            out = tr.update(D, DUMMY)
            for x1, y1, x2, y2, tid, *_ in out:
                b = (x1, y1, x2 - x1, y2 - y1)
                if any(b[0] > i[0] and b[1] > i[1] and b[0] + b[2] < i[0] + i[2] and b[1] + b[3] < i[1] + i[3] for i in igs[seq].get(f, [])):
                    continue
                pred.setdefault(f, []).append((int(tid), *b))
        acc = mm.MOTAccumulator(auto_id=True)
        for f in range(1, len(data[seq]["dets"]) + 1):
            g, p = gts[seq].get(f, []), pred.get(f, [])
            dist = iou_dist([x[1:] for x in g], [x[1:] for x in p]) if g and p else []
            acc.update([x[0] for x in g], [x[0] for x in p], dist)
        accs.append(acc)
    s = mm.metrics.create().compute_many(accs, metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"], generate_overall=False)
    return dict(MOTA=s["mota"].mean(), IDF1=s["idf1"].mean(), IDS=int(s["num_switches"].sum()),
                FP=int(s["num_false_positives"].sum()), FN=int(s["num_misses"].sum()))


log = []
def trial(stage, cfg, note=""):
    r = run(cfg); log.append(dict(stage=stage, **{k: cfg[k] for k in DEFAULT}, **r))
    print(f"[{stage}] {note:34s} MOTA={r['MOTA']:.3f} IDF1={r['IDF1']:.3f} IDS={r['IDS']:4d} FP={r['FP']:5d} FN={r['FN']:5d}", flush=True)
    return r

best = dict(DEFAULT); best_r = trial("base", best, "por defecto")
def search(stage, name, values, extra=None):
    global best, best_r
    cands = []
    for v in values:
        cfg = dict(best, **{name: v}, **(extra(v) if extra else {}))
        if cfg["low"] >= cfg["high"] + 1e-9 and name in ("low", "high"): continue
        cands.append((trial(stage, cfg, f"{name}={v}"), cfg))
    r, cfg = max(cands, key=lambda t: t[0]["MOTA"])
    if r["MOTA"] > best_r["MOTA"]: best, best_r = cfg, r
    print(f"   -> mejor {name} = {best[name]} (MOTA {best_r['MOTA']:.3f})\n", flush=True)

# 1) umbrales de confianza (el diagnostico mostro que por debajo de ~0.7 casi todo son falsos positivos)
search("1-high", "high", [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9], extra=lambda v: {"new": v})
search("1-low", "low", [0.1, 0.3, 0.5, 0.6, 0.7, 0.75])
search("1-new", "new", [best["high"], round(best["high"] + 0.05, 2), round(best["high"] + 0.1, 2)])
# 2) tracker
search("2-buffer", "buffer", [15, 30, 60, 90])
search("2-match", "match", [0.6, 0.7, 0.8, 0.9])
search("2-gmc", "gmc", [True, False])
# 3) heuristicas sobre las detecciones
search("3-min_size", "min_size", [0, 15, 18, 20, 22, 25])
search("3-van", "van", [True, False])
search("3-truck", "truck_conf", [0.0, 0.8, 0.9, 0.95, 1.01])
# 4) un segundo barrido corto de los umbrales ya con las heuristicas
search("4-high", "high", sorted({round(best["high"] + d, 2) for d in (-0.1, -0.05, 0, 0.05, 0.1)}), extra=lambda v: {"new": max(v, best["new"] if v < best["new"] else v)})
search("4-low", "low", sorted({round(best["low"] + d, 2) for d in (-0.15, -0.1, -0.05, 0, 0.05, 0.1)}))

df = pd.DataFrame(log); df.to_csv("results/tuning_method3.csv", index=False)
json.dump(dict(best=best, metrics=best_r, default=log[0], frames_per_seq=a.max_frames, seqs=a.seqs), open("results/tuning_method3_best.json", "w"), indent=1, default=float)
print("\nMEJOR:", best); print("MEJOR METRICAS:", best_r); print("BASE  METRICAS:", {k: log[0][k] for k in ("MOTA", "IDF1", "IDS", "FP", "FN")})
