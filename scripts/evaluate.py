"""Evalua uno o varios metodos (results/<nombre>/) contra el GT con motmetrics (IoU>=0.5, GT *_gt.txt con score=1).
Imprime una tabla comparativa global (MOTA, IDF1, IDS, FPS) y guarda results/<nombre>/metrics.csv por secuencia
y results/comparison.csv con todos los metodos.
Uso: python -I scripts/evaluate.py --root data/<raiz> --names method1_yolo11_sort method2_yolo26_bytetrack ..."""
import argparse, json
from pathlib import Path
import pandas as pd
import motmetrics as mm
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_utils import iou_dist, load, load_ignore

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data")
ap.add_argument("--names", nargs="+", required=True, help="carpetas en results/ a evaluar")
ap.add_argument("--seqs", nargs="*", help="evaluar solo estas secuencias (p.ej. las de validacion, no usadas para ajustar)")
ap.add_argument("--fps_from", nargs="*", default=[], metavar="NOMBRE=CARPETA",
                help="tomar FPS y ms/frame de otra carpeta de results/ (medida en GPU), p.ej. method3_tuned=first-5-jobs/method3_rtdetrm_botsort")
a = ap.parse_args()
FPS_FROM = dict(x.split("=", 1) for x in a.fps_from)
GT_DIR = Path(a.root) / "UAV-benchmark-MOTD_v1.0" / "GT"


def evaluate(name):
    res = Path("results") / name
    timing = json.loads((res / "timing.json").read_text())
    accs = []
    if a.seqs: timing = {k: v for k, v in timing.items() if k in a.seqs}
    for seq, t in timing.items():
        gt = load(GT_DIR / f"{seq}_gt.txt")  # ya contiene solo score=1 y no tiene columna de clase (todo es car/truck/bus)
        pr = load(res / f"{seq}.txt", load_ignore(GT_DIR / f"{seq}_gt_ignore.txt"))
        acc = mm.MOTAccumulator(auto_id=True)
        for f in range(1, t["frames"] + 1):
            g, p = gt.get(f, []), pr.get(f, [])
            dist = iou_dist([x[1:] for x in g], [x[1:] for x in p]) if g and p else []
            acc.update([x[0] for x in g], [x[0] for x in p], dist)
        accs.append(acc)
    mh = mm.metrics.create()
    summ = mh.compute_many(accs, names=list(timing), generate_overall=True,
                           metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses",
                                    "mostly_tracked", "mostly_lost"])
    # FPS y ms/frame solo cuentan las secuencias medidas en GPU (device != cpu): un CPU y un GPU no son comparables.
    # Secuencias de timing.json sin campo "device" se consideran de GPU (los primeros jobs).
    # Con --fps_from NOMBRE=CARPETA los tiempos se toman de otra corrida (medida en GPU) del mismo metodo.
    src = FPS_FROM.get(name)
    ftim = json.loads((Path("results") / src / "timing.json").read_text()) if src else timing
    gpu = {k: t for k, t in ftim.items() if t.get("device", "gpu") != "cpu"}
    nan = float("nan")
    def per(k, f): return f(gpu[k]) if k in gpu else nan
    fr = sum(t["frames"] for t in gpu.values())
    summ["fps"] = [per(k, lambda t: t["fps"]) for k in timing] + [fr / sum(t["frames"] / t["fps"] for t in gpu.values()) if fr else nan]
    summ["det_ms"] = [per(k, lambda t: 1e3 * t["t_det"] / t["frames"]) for k in timing] + [1e3 * sum(t["t_det"] for t in gpu.values()) / fr if fr else nan]
    summ["trk_ms"] = [per(k, lambda t: 1e3 * t["t_trk"] / t["frames"]) for k in timing] + [1e3 * sum(t["t_trk"] for t in gpu.values()) / fr if fr else nan]
    summ["n_gpu_seqs"] = [int(k in gpu) for k in timing] + [len(gpu)]
    # El PDF pide indicadores PROMEDIADOS sobre las secuencias: fila MEAN = promedio simple de las secuencias
    # (OVERALL = todos los frames juntos, pondera mas las secuencias largas). IDS/FP/FN se suman, no se promedian.
    per_seq = summ.drop(index="OVERALL")
    summ.loc["MEAN"] = per_seq.mean()
    for c in ("num_switches", "num_false_positives", "num_misses", "n_gpu_seqs"):
        summ.loc["MEAN", c] = per_seq[c].sum()
    summ.to_csv(res / "metrics.csv")
    return summ


overall, mean = {}, {}
for n in a.names:
    s = evaluate(n)
    print(f"\n== {n} ==")
    print(s.round(3).to_string())
    overall[n], mean[n] = s.loc["OVERALL"], s.loc["MEAN"]

cols = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses", "fps", "det_ms", "trk_ms", "n_gpu_seqs"]
names = ["MOTA", "IDF1", "IDS", "FP", "FN", "FPS", "det_ms/frame", "trk_ms/frame", "seqs_gpu"]
for title, d, f in (("PROMEDIO sobre secuencias (lo que pide el PDF)", mean, "comparison_mean.csv"),
                    ("GLOBAL (todos los frames juntos)", overall, "comparison_overall.csv")):
    cmp = pd.DataFrame(d).T[cols]; cmp.columns = names
    print(f"\n== COMPARACION: {title} =="); print(cmp.round(3).to_string())
    cmp.to_csv(Path("results") / f)
