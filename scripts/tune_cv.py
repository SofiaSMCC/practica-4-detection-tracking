"""Ajuste de hiperparametros de los 4 metodos con VALIDACION CRUZADA sobre detecciones cacheadas.

Por que: ajustar y evaluar sobre las mismas secuencias infla los numeros (nos paso con el metodo 3: 0.70 en las 5 de
ajuste y 0.22 en las 15 de validacion). Aqui las 20 secuencias de test se reparten en 4 pliegues; para cada pliegue se
ajusta con las otras 15 y se mide en las 5 no vistas. El promedio de los 4 pliegues es la estimacion honesta.
Despues se ajusta con las 20 y se escribe la corrida final (results/<metodo>_cv/) para evaluar con evaluate.py.

Metodos (todos con pesos preentrenados, solo se ajusta el tracker y el umbral del detector):
  m1  YOLOv11 + SORT          m2  YOLO26 + ByteTrack
  m3  RT-DETRv4 + BoT-SORT    m4  YOLO26 + SAHI + ByteTrack
Necesita: python -I scripts/cache_detections.py --detector {yolo11,yolo26,rtdetr,sahi,gmc}
Uso: python -I scripts/tune_cv.py --method m2 [--win 300] [--seqs ...]"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
import motmetrics as mm

sys.path.insert(0, str(Path(__file__).parent))
from common import Dets, find_sequences, tracker_args
from eval_utils import iou_dist, load, load_ignore

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["m1", "m2", "m3", "m4", "m5"])
ap.add_argument("--root", default="data"); ap.add_argument("--cache", default="results/_cache")
ap.add_argument("--seqs", nargs="*"); ap.add_argument("--win", type=int, default=300, help="frames por ventana de ajuste")
ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--no_final", action="store_true", help="no escribir la corrida final con todos los frames")
a = ap.parse_args()

DET = {"m1": "yolo11", "m2": "yolo26", "m3": "rtdetr", "m4": "sahi", "m5": "fuse"}[a.method]
OUTNAME = {"m1": "method1_yolo11_sort_cv", "m2": "method2_yolo26_bytetrack_cv", "m3": "method3_rtdetrm_botsort_cv",
           "m4": "method4_sahi_yolo26_bytetrack_cv", "m5": "method5_fusion_bytetrack_cv"}[a.method]
GT_DIR = Path(a.root) / "UAV-benchmark-MOTD_v1.0" / "GT"
seqs = [s for s, _ in find_sequences(a.root, a.seqs)]
cache = Path(a.cache)
DUMMY = np.zeros((2, 2, 3), np.uint8)

dets = {s: pickle.load(open(cache / DET / f"{s}.pkl", "rb")) for s in seqs}
warps = {s: pickle.load(open(cache / "gmc" / f"{s}.pkl", "rb"))["warps"] for s in seqs} if a.method == "m3" else {}
gts = {s: load(GT_DIR / f"{s}_gt.txt") for s in seqs}
igs = {s: load_ignore(GT_DIR / f"{s}_gt_ignore.txt") for s in seqs}
for s in seqs: assert len(dets[s]["dets"]) == max(gts[s]), f"{s}: el cache tiene {len(dets[s]['dets'])} frames y el GT {max(gts[s])}"

# unidades de ajuste = ventanas contiguas (inicio y mitad de cada secuencia); el tracker se reinicia en cada una
units = []
for s in seqs:
    n = len(dets[s]["dets"]); starts = [0] + ([n // 2] if n >= 2 * a.win else [])
    units += [(s, st, min(st + a.win, n)) for st in starts]
fold_of = {s: i % a.folds for i, s in enumerate(seqs)}      # reparto alternado: cada pliegue mezcla escenas

DEFAULT = {"m1": dict(conf=0.25, max_age=30, min_hits=3, iou=0.3),
           "m2": dict(high=0.25, low=0.1, new=0.25, buffer=30, match=0.8),
           "m3": dict(high=0.25, low=0.1, new=0.25, buffer=30, match=0.8, gmc=True),
           "m4": dict(high=0.25, low=0.1, new=0.25, buffer=30, match=0.8),
           "m5": dict(high=0.25, low=0.1, new=0.25, buffer=30, match=0.8)}[a.method]
GROUPS = {   # (parametro, valores a probar) ; "high" arrastra "new" cuando new == high
    "m1": [("conf", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), ("min_hits", [1, 2, 3, 5]), ("max_age", [10, 30, 60]), ("iou", [0.1, 0.2, 0.3, 0.4, 0.5]), ("conf", [0.15, 0.25, 0.35, 0.45, 0.55])],
}
common_groups = [("high", [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]), ("low", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]), ("new", None),
                 ("buffer", [15, 30, 60]), ("match", [0.7, 0.8, 0.9]), ("high", None), ("low", None)]
GROUPS["m2"] = GROUPS["m4"] = GROUPS["m5"] = common_groups
GROUPS["m3"] = common_groups[:5] + [("gmc", [True, False])] + common_groups[5:]

MAP3 = {3: 1, 4: 1, 5: 2, 8: 3}


def make_tracker(cfg):
    if a.method == "m1":
        from sort import Sort
        return Sort(cfg["max_age"], cfg["min_hits"], cfg["iou"])
    ta = tracker_args("bytetrack.yaml" if a.method != "m3" else "botsort.yaml", track_high_thresh=cfg["high"], track_low_thresh=cfg["low"],
                      new_track_thresh=cfg["new"], track_buffer=cfg["buffer"], match_thresh=cfg["match"])
    if a.method == "m3":
        from ultralytics.trackers.bot_sort import BOTSORT
        return BOTSORT(ta)
    from ultralytics.trackers.byte_tracker import BYTETracker
    return BYTETracker(ta)


class CachedGMC:
    method = "cached"
    def __init__(self, w, off): self.w, self.i, self.off = w, 0, off
    def apply(self, img, d=None):
        w = np.eye(2, 3) if self.off else self.w[self.i]; self.i += 1; return w
    def reset_params(self): self.i = 0


def track_unit(cfg, seq, st, en):
    """Corre el tracker sobre las detecciones cacheadas de [st, en). Devuelve {frame: [(id, l, t, w, h)]} (frames 1-based)."""
    tr = make_tracker(cfg)
    if a.method == "m3": tr.gmc = CachedGMC(warps[seq][st:en], not cfg["gmc"])
    pred = {}
    for k in range(st, en):
        d = dets[seq]["dets"][k]
        if len(d):
            thr = cfg["conf"] if a.method == "m1" else cfg["low"]
            keep = d[:, 4] >= thr
            if a.method == "m3": keep &= np.isin(d[:, 5].astype(int), list(MAP3))
            d = d[keep]
        cls = np.array([MAP3.get(int(c), 0) for c in d[:, 5]]) if a.method == "m3" else d[:, 5] + 1
        if a.method == "m1":
            out = tr.update(np.column_stack([d[:, :4], d[:, 4], cls]) if len(d) else np.zeros((0, 6)))   # (M,7)
        else:
            out = tr.update(Dets(d[:, :4], d[:, 4], cls), DUMMY)
        for r in out:
            x1, y1, x2, y2, tid = r[:5]
            b = (x1, y1, x2 - x1, y2 - y1); f = k + 1
            if any(b[0] > i[0] and b[1] > i[1] and b[0] + b[2] < i[0] + i[2] and b[1] + b[3] < i[1] + i[3] for i in igs[seq].get(f, [])):
                continue
            pred.setdefault(f, []).append((int(tid), *b))
    return pred


def score_unit(pred, seq, st, en):
    acc = mm.MOTAccumulator(auto_id=True)
    for f in range(st + 1, en + 1):
        g, p = gts[seq].get(f, []), pred.get(f, [])
        dist = iou_dist([x[1:] for x in g], [x[1:] for x in p]) if g and p else []
        acc.update([x[0] for x in g], [x[0] for x in p], dist)
    return acc


memo = {}
def unit_scores(cfg):
    """Metricas por unidad para una configuracion (memorizado: no depende del pliegue)."""
    key = tuple(sorted(cfg.items()))
    if key not in memo:
        accs = [score_unit(track_unit(cfg, s, st, en), s, st, en) for s, st, en in units]
        df = mm.metrics.create().compute_many(accs, metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"],
                                              names=[f"{s}_{st}" for s, st, _ in units], generate_overall=False)
        memo[key] = {u: df.iloc[i].to_dict() for i, u in enumerate(units)}
    return memo[key]


def score(cfg, subset):
    """Promedio sobre secuencias (de promedio sobre sus ventanas) de MOTA/IDF1; IDS/FP/FN sumados."""
    us = unit_scores(cfg); out = {}
    for m in ("mota", "idf1"):
        out[m] = float(np.mean([np.mean([us[u][m] for u in units if u[0] == s]) for s in subset]))
    for m, k in (("num_switches", "IDS"), ("num_false_positives", "FP"), ("num_misses", "FN")):
        out[k] = int(sum(us[u][m] for u in units if u[0] in subset))
    return out


def descend(train):
    best, best_r = dict(DEFAULT), score(DEFAULT, train)
    for name, values in GROUPS[a.method]:
        if name == "new" and values is None: values = sorted({best["high"], round(best["high"] + 0.05, 2), round(best["high"] + 0.1, 2)})
        if values is None:   # segunda pasada fina alrededor del valor actual
            values = sorted({round(min(max(best[name] + d, 0.05), 0.95), 2) for d in (-0.1, -0.05, 0, 0.05, 0.1)})
        for v in values:
            cfg = dict(best, **{name: v})
            if name == "high": cfg["new"] = max(v, cfg["new"]) if best["new"] > best["high"] else v
            if a.method != "m1" and cfg["low"] >= cfg["high"]: continue
            r = score(cfg, train)
            if r["mota"] > best_r["mota"] + 1e-9: best, best_r = cfg, r
    return best, best_r


t0 = time.time(); report = {"method": a.method, "detector": DET, "units": len(units), "win": a.win, "folds": []}
folds = sorted(set(fold_of.values()))
held, held_def = [], []
for k in folds:
    test = [s for s in seqs if fold_of[s] == k]; train = [s for s in seqs if fold_of[s] != k]
    cfg, r_train = descend(train)
    r_test, r_def = score(cfg, test), score(DEFAULT, test)
    held.append(r_test["mota"]); held_def.append(r_def["mota"])
    report["folds"].append(dict(fold=k, test=test, cfg=cfg, train=r_train, heldout=r_test, heldout_default=r_def))
    print(f"[{a.method}] pliegue {k}: ajustado MOTA(entrenamiento)={r_train['mota']:.3f} | validacion={r_test['mota']:.3f} "
          f"(por defecto {r_def['mota']:.3f}) cfg={ {x: cfg[x] for x in cfg if cfg[x] != DEFAULT[x]} } [{time.time()-t0:.0f}s]", flush=True)
report["cv_mota"] = float(np.mean(held)); report["cv_mota_default"] = float(np.mean(held_def))
print(f"[{a.method}] VALIDACION CRUZADA: MOTA ajustado={report['cv_mota']:.3f} frente a por defecto={report['cv_mota_default']:.3f}", flush=True)

final, r_final = descend(seqs)
report["final"] = dict(cfg=final, in_sample=r_final, default_in_sample=score(DEFAULT, seqs))
print(f"[{a.method}] FINAL (20 sec., con muestra): {final}\n   MOTA {r_final['mota']:.3f} (por defecto {report['final']['default_in_sample']['mota']:.3f})", flush=True)
Path("results/tuning_cv").mkdir(parents=True, exist_ok=True)
json.dump(report, open(f"results/tuning_cv/{a.method}.json", "w"), indent=1, default=float)

if not a.no_final:   # corrida final con TODOS los frames y formato MOT estandar (para evaluate.py y las demos)
    out = Path("results") / OUTNAME; out.mkdir(parents=True, exist_ok=True); timing = {}
    for s in seqs:
        n = len(dets[s]["dets"]); pred = track_unit(final, s, 0, n)
        lines = [f"{f},{i},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1" for f in sorted(pred) for (i, x, y, w, h) in pred[f]]
        (out / f"{s}.txt").write_text("\n".join(lines) + "\n")
        timing[s] = dict(frames=n, t_det=dets[s]["t_det"], t_trk=0.0, fps=n / max(dets[s]["t_det"], 1e-9), device="cpu")
    (out / "timing.json").write_text(json.dumps(timing, indent=1)); print("escrito", out)
