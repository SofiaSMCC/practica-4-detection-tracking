"""Analisis de los falsos positivos persistentes del metodo 4 ANTES del filtro de escala (motiva esa heuristica).
Para cada secuencia: corre la fusion + ByteTrack (configuracion de E1, sin post-proceso), separa los tracks en
verdaderos (> 50 % de sus cajas emparejadas con el GT, IoU >= 0.5) y falsos, y mide
  - lado tipico de las cajas FP frente al de los vehiculos del GT (sqrt(w*h), mediana)
  - % de frames FP en tracks 'estaticos' (desplazamiento compensado por GMC < 10 px entre su primer y ultimo frame)
Salida: results/m4_exp/fp_analysis.json.  Uso: python -I scripts/m4_fp_analysis.py [--variant YRS_slot] [--seqs ...]"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m4_lib as L
from eval_utils import iou_matrix, load, load_ignore

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="YRS_slot"); ap.add_argument("--root", default="data")
ap.add_argument("--cache", default="results/_cache"); ap.add_argument("--seqs", nargs="*")
a = ap.parse_args()
GT = Path(a.root) / "UAV-benchmark-MOTD_v1.0" / "GT"
tcfg = json.load(open("results/m4_exp/fusion_cv.json"))[a.variant]["final"]["cfg"][0]
seqs = a.seqs or sorted(p.stem for p in (Path(a.cache) / f"fx_{a.variant}").glob("*.pkl"))
out = {"variant": a.variant, "tracker": tcfg, "seqs": {}}
for seq in seqs:
    fx = pickle.load(open(Path(a.cache) / f"fx_{a.variant}" / f"{seq}.pkl", "rb"))["dets"]
    A = L.cum_affine(pickle.load(open(Path(a.cache) / "gmc" / f"{seq}.pkl", "rb"))["warps"])
    gt, ig = load(GT / f"{seq}_gt.txt"), load_ignore(GT / f"{seq}_gt_ignore.txt")
    tr = L.run_bytetrack(fx, tcfg)
    fp_side, fp_frames, fp_static = [], 0, 0
    for tid, obs in tr.items():
        fr = sorted(obs); hit = tot = 0; sides = []
        for f in fr:
            x1, y1, x2, y2, _ = obs[f]; b = (x1, y1, x2 - x1, y2 - y1)
            if L.inside_ignore(b, ig.get(f, [])): continue
            tot += 1; sides.append(np.sqrt(max(b[2], 0) * max(b[3], 0))); g = gt.get(f, [])
            hit += bool(g) and iou_matrix([x[1:] for x in g], [b]).max() >= 0.5
        if not tot or hit / tot > 0.5: continue
        (a1, b1, c1, d1, _), (a2, b2, c2, d2, _) = obs[fr[0]], obs[fr[-1]]
        p = (A[fr[-1] - 1] @ np.linalg.inv(A[fr[0] - 1])) @ np.array([(a1 + c1) / 2, (b1 + d1) / 2, 1.0])
        static = np.linalg.norm(p[:2] - [(a2 + c2) / 2, (b2 + d2) / 2]) < 10
        fp_side += sides; fp_frames += tot; fp_static += tot * static
    gt_side = [np.sqrt(x[3] * x[4]) for v in gt.values() for x in v]
    out["seqs"][seq] = dict(fp_side=float(np.median(fp_side)) if fp_side else None, gt_side=float(np.median(gt_side)),
                            fp_frames=int(fp_frames), static_pct=100.0 * fp_static / max(fp_frames, 1))
    print(seq, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in out["seqs"][seq].items()}, flush=True)
json.dump(out, open("results/m4_exp/fp_analysis.json", "w"), indent=1)
