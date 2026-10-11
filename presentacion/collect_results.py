"""Junta en presentacion/results.json todos los numeros que usa la presentacion (lee results/*.csv|json). 
Uso: python -I presentacion/collect_results.py   (desde la raiz del proyecto)"""
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "scripts")
from eval_utils import iou_matrix, load, load_ignore

R = Path("results"); GT = Path("data/UAV-benchmark-MOTD_v1.0/GT")
rd = lambda f: {r[""]: r for r in csv.DictReader(open(f))}
fl = lambda x: float(x) if x not in ("", "nan") else None
M = {  # clave: (carpeta ajustada, carpeta por defecto, metodo de ajuste, carpeta con FPS de GPU)
    "m1": ("method1_yolo11_sort_cv", "method1_yolo11_sort", "m1"),
    "m2": ("method2_yolo26_bytetrack_cv", "method2_yolo26_bytetrack", "m2"),
    "m3": ("method3_rtdetrm_botsort_cv", "method3_rtdetrm_botsort_tuned", "m3"),
    "m4": ("method4_sahi_yolo26_bytetrack_cv", "method4_sahi_yolo26_bytetrack", "m4"),
    "m5": ("method5_fusion_bytetrack_cv", None, "m5"),
}
mean, overall = rd(R / "final_cv5_20_mean.csv"), rd(R / "final_cv5_20_overall.csv")
dmean, doverall = rd(R / "final_20_mean.csv"), rd(R / "final_20_overall.csv")
GPU_FPS = {"m1": 39.03, "m2": 38.98, "m3": 17.33, "m4": 9.10, "m2nms": 32.18}   # corridas GPU de las 5 primeras secuencias (valores por defecto)
GPU_DET = {"m1": 26.15, "m2": 24.61, "m3": 50.57, "m4": 110.17, "m2nms": 30.43}
GPU_TRK = {"m1": 0.84, "m2": 1.14, "m3": 7.17, "m4": 1.25, "m2nms": 1.15}

seqs = sorted(s for s in rd(R / "method2_yolo26_bytetrack_cv/metrics.csv") if s.startswith("M") and s[1:].isdigit())
out = {"seqs": seqs, "methods": {}, "per_seq": {}, "cv": {}, "cfg": {}}
for k, (cv, df, tk) in M.items():
    j = json.load(open(R / "tuning_cv" / f"{tk}.json"))
    out["cv"][k] = dict(tuned=j["cv_mota"], default=j["cv_mota_default"], folds=[(f["heldout"]["mota"], f["heldout_default"]["mota"]) for f in j["folds"]])
    out["cfg"][k] = j["final"]["cfg"]
    m, o = mean[cv], overall[cv]
    out["methods"][k] = dict(mota=fl(m["MOTA"]), mota_g=fl(o["MOTA"]), idf1=fl(m["IDF1"]), idf1_g=fl(o["IDF1"]), ids=fl(m["IDS"]), fp=fl(m["FP"]), fn=fl(m["FN"]),
                             mota_default=fl(dmean[df]["MOTA"]) if df else None, idf1_default=fl(dmean[df]["IDF1"]) if df else None,
                             ids_default=fl(dmean[df]["IDS"]) if df else None, fps=GPU_FPS.get(k))
    met = rd(R / cv / "metrics.csv"); out["per_seq"][k] = {s: fl(met[s]["mota"]) for s in seqs}
out["methods"]["m5"]["fps"] = round(1000 / (GPU_DET["m2"] + GPU_DET["m3"] + GPU_TRK["m2"]), 1)   # estimado
out["methods"]["m5"]["fps_estimado"] = True
# metodo 1 ajustado/por defecto y NMS (valores por defecto, 20 secuencias)
out["nms"] = {k: dict(mota=fl(dmean[d]["MOTA"]), idf1=fl(dmean[d]["IDF1"]), ids=fl(dmean[d]["IDS"]), det_ms=GPU_DET[x], trk_ms=GPU_TRK[x], fps=GPU_FPS[x])
              for k, d, x in (("sin_nms", "method2_yolo26_bytetrack", "m2"), ("con_nms", "method2_yolo26_nms_bytetrack", "m2nms"))}

# recall por tamano (terciles del GT de las 20 secuencias), con y sin ajuste
def recall(name):
    sizes = {s: np.array([np.sqrt(float(l.split(",")[4]) * float(l.split(",")[5])) for l in (GT / f"{s}_gt.txt").read_text().split()]) for s in seqs}
    cut = np.percentile(np.concatenate(list(sizes.values())), [33.3, 66.7]); hit = np.zeros(3); tot = np.zeros(3)
    for s in seqs:
        gt = load(GT / f"{s}_gt.txt"); pr = load(R / name / f"{s}.txt", load_ignore(GT / f"{s}_gt_ignore.txt"))
        for f in range(1, max(gt) + 1):
            g, p = [x[1:] for x in gt.get(f, [])], [x[1:] for x in pr.get(f, [])]
            ok = set()
            if g and p:
                iou = iou_matrix(g, p); a, b = linear_sum_assignment(-iou); ok = {i for i, j in zip(a, b) if iou[i, j] >= 0.5}
            for i, x in enumerate(g):
                sz = np.sqrt(x[2] * x[3]); c = 0 if sz < cut[0] else 1 if sz < cut[1] else 2
                tot[c] += 1; hit[c] += i in ok
    return (hit / tot).round(3).tolist(), [round(float(c), 1) for c in cut]
out["recall"] = {}
for k, (cv, df, _) in M.items():
    out["recall"][k] = {"tuned": recall(cv)[0]}
    if df: out["recall"][k]["default"] = recall(df)[0]
out["recall_cut"] = recall(M["m2"][0])[1]
out["sahi_compare"] = json.load(open(R / "sahi_compare.json"))
# MOTA promedio por tipo de escena (atributos de M_attr: dia, noche, niebla, altura, vista, larga duracion)
names = ["día", "noche", "niebla", "baja altura", "media altura", "gran altura", "vista frontal", "vista lateral", "vista cenital", "larga duración"]
attr = {}
for s in seqs:
    f = Path(f"data/M_attr/test/{s}_attr.txt")
    if not f.exists(): f = next(Path("data/M_attr/test").glob(f"{s}*_attr.txt"))
    attr[s] = [int(x) for x in f.read_text().strip().split(",")]
out["groups"] = {}
for j, n in enumerate(names):
    g = [s for s in seqs if attr[s][j]]
    if g: out["groups"][n] = dict(n=len(g), **{k: float(np.mean([out["per_seq"][k][s] for s in g])) for k in out["per_seq"]})
json.dump(out, open("presentacion/results.json", "w"), indent=1, default=float)
print(json.dumps({k: out["methods"][k] for k in out["methods"]}, indent=0, default=float)[:900]); print("cortes:", out["recall_cut"])
