"""Agrega a presentacion/results.json los numeros del METODO 4 FINAL (SAHI + fusion + filtro de escala), bajo la clave "m4n".
Lee results/method4_sahi_fusion_bytetrack/metrics.csv, results/m4_exp/{ablation_m4.csv, size_recall_m4.csv,
post_cv_YRS_slot.json, post_final_YRS_slot.json, fp_analysis.json}. Correr DESPUES de collect_results.py:
    python -I presentacion/collect_results.py && python -I presentacion/collect_m4.py && node presentacion/build_deck.js
Los FPS vienen de la corrida end-to-end en GPU de Khipu (scripts/method4_final.py, jobs 56265/56266)."""
import csv, json
from pathlib import Path
import numpy as np

R, OUT = Path("results"), Path("presentacion/results.json")
rs = json.load(open(OUT))
rd = lambda f: {r[""]: r for r in csv.DictReader(open(f))}
met = rd(R / "method4_sahi_fusion_bytetrack" / "metrics.csv")
post = json.load(open(R / "m4_exp" / "post_cv_YRS_slot.json"))
final = json.load(open(R / "m4_exp" / "post_final_YRS_slot.json"))["final"]["cfg"]
abl = list(csv.DictReader(open(R / "m4_exp" / "ablation_m4.csv")))
rec = {r["metodo"]: r for r in csv.DictReader(open(R / "m4_exp" / "size_recall_m4.csv"))}
fpa = json.load(open(R / "m4_exp" / "fp_analysis.json"))["seqs"]
GPU = dict(fps=11.909, det_ms=82.778, trk_ms=1.194, fps_20=11.709)        # evaluate.py sobre method4_final_YRS_slot_gpu (5 / 20 sec.)

m, o, seqs = met["MEAN"], met["OVERALL"], rs["seqs"]
rs["methods"]["m4n"] = dict(mota=float(m["mota"]), mota_g=float(o["mota"]), idf1=float(m["idf1"]), idf1_g=float(o["idf1"]),
                            ids=float(m["num_switches"]), fp=float(m["num_false_positives"]), fn=float(m["num_misses"]), fps=GPU["fps"])
rs["per_seq"]["m4n"] = {s: float(met[s]["mota"]) for s in seqs}
best = post["ablation"]["escala + largo"]
rs["cv"]["m4n"] = dict(tuned=best["cv_mota"], default=None, protocol="secuencias completas, ByteTrack fijo (E1) + grilla de post-proceso")
rs["cfg"]["m4n"] = dict(tracker=final[0], post={k: final[1][k] for k in ("rel_size", "min_len")})
rs["gpu_m4n"] = GPU
rs["ablation_m4"] = [dict(paso=r["paso"], cfg=r["configuracion"], cv_mota=float(r["cv_mota"]), cv_idf1=float(r["cv_idf1"]),
                          ids=int(r["IDS"]), fp=int(r["FP"]), fn=int(r["FN"])) for r in abl]
rs["recall"]["m4n"] = {"tuned": [round(float(rec["method4_sahi_fusion_bytetrack"][f"recall_{t}"]), 3) for t in ("pequeno", "mediano", "grande")]}
rs["recall"]["m4n_sin_sahi"] = {"tuned": [round(float(rec["ablation_m4_sin_sahi"][f"recall_{t}"]), 3) for t in ("pequeno", "mediano", "grande")]}
tot = sum(v["fp_frames"] for v in fpa.values())
rs["fp_analysis"] = dict(seqs={s: fpa[s] for s in ("M1101", "M1007", "M0601", "M0701")},
                         static_pct_total=100 * sum(v["fp_frames"] * v["static_pct"] / 100 for v in fpa.values()) / tot)
# MOTA por tipo de escena. Si data/M_attr existe se recalcula; si no, se recupera que secuencias forman cada grupo a partir
# de los promedios ya guardados (busqueda exhaustiva: cada grupo tiene una unica combinacion de secuencias que los reproduce)
attr_dir = Path("data/M_attr/test")
names = list(rs["groups"])
if attr_dir.exists():
    allnames = ["día", "noche", "niebla", "baja altura", "media altura", "gran altura", "vista frontal", "vista lateral", "vista cenital", "larga duración"]
    attr = {s: [int(x) for x in next(attr_dir.glob(f"{s}*_attr.txt")).read_text().strip().split(",")] for s in seqs}
    memb = {n: [s for s in seqs if attr[s][allnames.index(n)]] for n in names}
else:
    import itertools
    X = np.array([[rs["per_seq"][k][s] for k in ("m1", "m2", "m3", "m4")] for s in seqs]); memb = {}
    for n, g in rs["groups"].items():
        tgt = np.array([g[k] for k in ("m1", "m2", "m3", "m4")])
        memb[n] = next([seqs[i] for i in c] for c in itertools.combinations(range(len(seqs)), g["n"]) if np.abs(X[list(c)].mean(0) - tgt).max() < 1e-9)
for n in names: rs["groups"][n]["m4n"] = float(np.mean([rs["per_seq"]["m4n"][s] for s in memb[n]]))
json.dump(rs, open(OUT, "w"), indent=1, default=float)
print("m4n:", {k: round(v, 3) for k, v in rs["methods"]["m4n"].items()}, "| CV", round(rs["cv"]["m4n"]["tuned"], 3))
