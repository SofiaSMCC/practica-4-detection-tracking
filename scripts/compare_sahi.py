"""Compara YOLO26 solo, SAHI 640 y SAHI 320 sobre las secuencias con vehiculos pequenos (primeros N frames).
Mismo tracker (ByteTrack) y mismos parametros (los ajustados del metodo 4) para los tres, asi solo cambia el detector.
Uso: python -I scripts/compare_sahi.py [--frames 300]"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import motmetrics as mm
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).parent))
from common import Dets, tracker_args
from eval_utils import iou_dist, iou_matrix, load, load_ignore
from ultralytics.trackers.byte_tracker import BYTETracker

ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, default=300)
ap.add_argument("--seqs", nargs="*", default=["M0602", "M0601", "M1301", "M1004", "M1302"]); a = ap.parse_args()
GT = Path("data/UAV-benchmark-MOTD_v1.0/GT"); C = Path("results/_cache")
cfg = json.load(open("results/tuning_cv/m4.json"))["final"]["cfg"]
ta = tracker_args("bytetrack.yaml", track_high_thresh=cfg["high"], track_low_thresh=cfg["low"], new_track_thresh=cfg["new"],
                  track_buffer=cfg["buffer"], match_thresh=cfg["match"])
CUT = (18.3, 28.4)
res = {}
for name, det in (("YOLO26 solo", "yolo26"), ("SAHI 640 / 0,2", "sahi"), ("SAHI 320 / 0,2", "sahi320")):
    accs, hit, tot, fp = [], np.zeros(3), np.zeros(3), 0
    for s in a.seqs:
        d = pickle.load(open(C / det / f"{s}.pkl", "rb"))["dets"][: a.frames]
        gt, ign = load(GT / f"{s}_gt.txt"), load_ignore(GT / f"{s}_gt_ignore.txt")
        tr = BYTETracker(ta); pred = {}
        for k, dd in enumerate(d):
            dd = dd[dd[:, 4] >= cfg["low"]] if len(dd) else dd
            out = tr.update(Dets(dd[:, :4], dd[:, 4], dd[:, 5] + 1) if len(dd) else Dets(np.zeros((0, 4)), [], []), np.zeros((2, 2, 3), np.uint8))
            for r in out:
                b = (r[0], r[1], r[2] - r[0], r[3] - r[1])
                if any(b[0] > i[0] and b[1] > i[1] and b[0] + b[2] < i[0] + i[2] and b[1] + b[3] < i[1] + i[3] for i in ign.get(k + 1, [])): continue
                pred.setdefault(k + 1, []).append((int(r[4]), *b))
        acc = mm.MOTAccumulator(auto_id=True)
        for f in range(1, len(d) + 1):
            g, p = gt.get(f, []), pred.get(f, [])
            acc.update([x[0] for x in g], [x[0] for x in p], iou_dist([x[1:] for x in g], [x[1:] for x in p]) if g and p else [])
            if g and p:
                iou = iou_matrix([x[1:] for x in g], [x[1:] for x in p]); r_, c_ = linear_sum_assignment(-iou)
                ok = {i for i, j in zip(r_, c_) if iou[i, j] >= 0.5}; fp += len(p) - len(ok)
            else:
                ok = set(); fp += len(p)
            for i, x in enumerate(g):
                sz = np.sqrt(x[3] * x[4]); b = 0 if sz < CUT[0] else 1 if sz < CUT[1] else 2
                tot[b] += 1; hit[b] += i in ok
        accs.append(acc)
    s_ = mm.metrics.create().compute_many(accs, metrics=["mota", "idf1", "num_switches", "num_misses"], generate_overall=False)
    res[name] = dict(MOTA=float(s_["mota"].mean()), IDF1=float(s_["idf1"].mean()), IDS=int(s_["num_switches"].sum()), FN=int(s_["num_misses"].sum()),
                     FP=int(fp), recall=(hit / tot).round(3).tolist(), n=tot.astype(int).tolist())
    print(f"{name:16s} MOTA={res[name]['MOTA']:.3f} IDF1={res[name]['IDF1']:.3f} FN={res[name]['FN']:6d} FP={res[name]['FP']:6d} "
          f"recall pequeño/mediano/grande={res[name]['recall']}", flush=True)
json.dump(res, open("results/sahi_compare.json", "w"), indent=1)
