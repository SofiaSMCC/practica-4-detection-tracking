"""Experimentos del METODO 4 = SAHI + heuristicas (fusion de detectores + post-proceso de tracks) con VALIDACION CRUZADA.

Todo corre en CPU sobre detecciones cacheadas (no hace falta GPU ni imagenes; solo el GT y results/_cache/).
Mismo protocolo que tune_cv.py: 20 secuencias de test en 4 pliegues alternados (i % 4); en cada pliegue se elige la
configuracion con las 15 secuencias de ajuste y se mide en las 5 no vistas. El promedio de los 4 pliegues es la
estimacion honesta. Para ajustar se usan ventanas de 300 frames (inicio y mitad de cada secuencia), como tune_cv.py.

Etapas (results/m4_exp/):
  prefuse precalcula las detecciones fusionadas de cada variante en results/_cache/fx_<variante>/ (no necesita GT)
  fusion  E1: que detectores fusionar, con que pesos y umbral IoU (barre tambien el umbral alto de ByteTrack, porque
          la escala de confianza cambia con la fusion). Incluye 'S' (SAHI solo = metodo 4 anterior) e 'YR' (= metodo 5).
  post    E2/E3 (la que se usa): ByteTrack FIJO (el de E1) + grilla de post-proceso sobre SECUENCIAS COMPLETAS,
          seleccion por validacion cruzada y ablacion por componente  -> post_cv_<variante>.json, post_final_<variante>.json
  tune    (descartada: sobreajusta) descenso por coordenadas sobre ByteTrack + post-proceso para una variante de fusion, con ablacion
          --post none | filter | scale | interp | static | all.
  final   corrida con TODOS los frames de las 20 secuencias -> results/<--name>/<seq>.txt + timing.json (para evaluate.py)

Uso:
  python -I scripts/m4_experiments.py fusion [--jobs 2]
  python -I scripts/m4_experiments.py post --variants YR YRS_slot
  python -I scripts/m4_experiments.py final --from results/m4_exp/post_final_YR.json --name method4_sahi_fusion_bytetrack
"""
import argparse, itertools, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m4_lib as L
from eval_utils import load, load_ignore

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("stage", choices=["prefuse", "fusion", "post", "tune", "final"])
ap.add_argument("--root", default="data"); ap.add_argument("--cache", default="results/_cache")
ap.add_argument("--out", default="results/m4_exp"); ap.add_argument("--seqs", nargs="*")
ap.add_argument("--win", type=int, default=300); ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--jobs", type=int, default=2)
ap.add_argument("--variants", nargs="*", help="solo estas variantes de fusion (etapa fusion)")
ap.add_argument("--variant", default="auto", help="variante de fusion para tune (auto = la mejor de la etapa fusion)")
ap.add_argument("--post", default="all", choices=["none", "filter", "scale", "interp", "static", "all"])
ap.add_argument("--from", dest="cfg_from", help="json de tune con la configuracion final (etapa final)")
ap.add_argument("--name", default="method4_sahi_fusion_bytetrack")
a = ap.parse_args()

VARIANTS = L.VARIANTS          # variantes de fusion (E1), definidas en m4_lib.py
DEFAULT_TRK = dict(high=0.25, low=0.1, new=0.25, buffer=30, match=0.8)   # ultralytics bytetrack.yaml
DEFAULT_POST = dict(min_len=0, min_score=0, rel_size=0, static_px=0, static_min=30, static_score=1.0, interp=0)

# ----------------------------------------------------------------------------- datos
GT_DIR = Path(a.root) / "UAV-benchmark-MOTD_v1.0" / "GT"
seqs = sorted(p.stem for p in (Path(a.cache) / "yolo26").glob("*.pkl"))
if a.seqs: seqs = [s for s in seqs if s in a.seqs]
if a.stage != "prefuse":
    assert seqs and all((GT_DIR / f"{s}_gt.txt").exists() for s in seqs), f"falta el GT en {GT_DIR}"
fold_of = {s: i % a.folds for i, s in enumerate(seqs)}
out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
G = {}   # datos por proceso (se cargan una vez en cada worker)


def data(seq):
    if seq not in G:
        G[seq] = dict(gt=load(GT_DIR / f"{seq}_gt.txt"), ig=load_ignore(GT_DIR / f"{seq}_gt_ignore.txt"))
    return G[seq]


def fused(variant, seq):
    """Detecciones fusionadas de una variante (se guardan en results/_cache/fx_<variante>/ para no recalcular)."""
    p = Path(a.cache) / f"fx_{variant}" / f"{seq}.pkl"
    if p.exists():
        import pickle; return pickle.load(open(p, "rb"))
    dets, ws, thr, *slots = VARIANTS[variant]
    loaded = [L.load_cache(a.cache, d, seq) for d in dets]
    fr = [x[0] for x in loaded]
    res = dict(dets=fr[0] if len(fr) == 1 else L.fuse_sequence(fr, ws, thr, slots[0] if slots else None),
               t_det=sum(x[1] for x in loaded))
    res["frames"] = len(res["dets"])
    import pickle, os
    p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(".tmp"); pickle.dump(res, open(tmp, "wb")); os.replace(tmp, p)
    return res


def units_of(seq, n):
    starts = [0] + ([n // 2] if n >= 2 * a.win else [])
    return [(st, min(st + a.win, n)) for st in starts]


def gmc_A(seq):
    import pickle
    return L.cum_affine(pickle.load(open(Path(a.cache) / "gmc" / f"{seq}.pkl", "rb"))["warps"])


def eval_seq(job):
    """(variante, seq, lista de (cfg_trk, lista de cfg_post)) -> lista de filas de metricas por ventana.
    El tracking se hace una vez por cfg_trk y se reutiliza para todos los cfg_post."""
    variant, seq, plan = job
    d = data(seq); fx = fused(variant, seq)["dets"]; n = len(fx)
    A = gmc_A(seq) if any(p.get("static_px", 0) for _, ps in plan for p in ps) else None
    rows = []
    for tcfg, pcfgs in plan:
        for st, en in units_of(seq, n):
            tr = L.run_bytetrack(fx, tcfg, st, en)
            accs, keys = [], []
            for pcfg in pcfgs:
                accs.append(L.accumulate(L.tracks_to_pred(L.postprocess(tr, pcfg, A), d["ig"]), d["gt"], st, en))
                keys.append(json.dumps([tcfg, pcfg], sort_keys=True))
            df = L.compute(accs, [str(i) for i in range(len(accs))])
            for i, k in enumerate(keys):
                r = df.iloc[i].to_dict(); rows.append(dict(key=k, seq=seq, st=st, **r))
    return rows


def aggregate(rows, subset):
    """MOTA/IDF1: promedio sobre secuencias del promedio de sus ventanas; IDS/FP/FN sumados (como tune_cv.py)."""
    by = {}
    for r in rows:
        if r["seq"] in subset: by.setdefault(r["key"], []).append(r)
    out = {}
    for k, rs in by.items():
        per = {}
        for r in rs: per.setdefault(r["seq"], []).append(r)
        if len(per) < len(subset): continue
        out[k] = dict(mota=float(np.mean([np.mean([x["mota"] for x in v]) for v in per.values()])),
                      idf1=float(np.mean([np.mean([x["idf1"] for x in v]) for v in per.values()])),
                      IDS=int(sum(r["num_switches"] for r in rs)), FP=int(sum(r["num_false_positives"] for r in rs)),
                      FN=int(sum(r["num_misses"] for r in rs)))
    return out


def run_jobs(jobs):
    with Pool(a.jobs) as pool:
        return [r for rs in pool.imap_unordered(eval_seq, jobs) for r in rs]


def cv_select(rows, keys, label):
    """Para cada pliegue: mejor key en ajuste, medida en las no vistas."""
    rep = []
    for k in range(a.folds):
        test = [s for s in seqs if fold_of[s] == k]; train = [s for s in seqs if fold_of[s] != k]
        tr = aggregate(rows, train); te = aggregate(rows, test)
        best = max((x for x in keys if x in tr), key=lambda x: tr[x]["mota"])
        rep.append(dict(fold=k, test=test, best=json.loads(best), train=tr[best], heldout=te[best]))
        print(f"[{label}] pliegue {k}: ajuste={tr[best]['mota']:.3f} validacion={te[best]['mota']:.3f} {best}", flush=True)
    cv = float(np.mean([r["heldout"]["mota"] for r in rep])); cv_idf1 = float(np.mean([r["heldout"]["idf1"] for r in rep]))
    allr = aggregate(rows, seqs); best_all = max((x for x in keys if x in allr), key=lambda x: allr[x]["mota"])
    print(f"[{label}] VALIDACION CRUZADA MOTA={cv:.3f} IDF1={cv_idf1:.3f} | final (20 sec.) {allr[best_all]['mota']:.3f} {best_all}", flush=True)
    return dict(folds=rep, cv_mota=cv, cv_idf1=cv_idf1, final=dict(cfg=json.loads(best_all), in_sample=allr[best_all]))


# ============================================================================= etapa E1: fusion
def stage_fusion():
    names = a.variants or list(VARIANTS)
    highs = [0.35, 0.45, 0.55, 0.65, 0.75]
    plan = [(dict(DEFAULT_TRK, high=h, new=h, low=0.2 if h > 0.25 else 0.1), [DEFAULT_POST]) for h in highs]
    t0 = time.time(); fp = out_dir / "fusion_cv.json"
    report = json.load(open(fp)) if fp.exists() else {}     # acumula: se pueden correr variantes por tandas
    for v in names:
        rows = run_jobs([(v, s, plan) for s in seqs])
        json.dump(rows, open(out_dir / f"fusion_rows_{v}.json", "w"))
        keys = sorted({r["key"] for r in rows})
        report[v] = cv_select(rows, keys, f"fusion {v}")
        report[v]["by_high"] = {json.loads(k)[0]["high"]: m for k, m in aggregate(rows, seqs).items()}
        print(f"   [{time.time() - t0:.0f}s]", flush=True)
        json.dump(report, open(out_dir / "fusion_cv.json", "w"), indent=1)
    print("\n== E1 resumen (MOTA en validacion cruzada) ==")
    for v, r in sorted(report.items(), key=lambda t: -t[1]["cv_mota"]):
        print(f"{v:10s} CV MOTA {r['cv_mota']:.3f}  IDF1 {r['cv_idf1']:.3f}  | 20 sec. {r['final']['in_sample']['mota']:.3f}")



# ============================================================================= etapa E2/E3 robusta: post-proceso en secuencias completas
PGRID = dict(rel_size=[0, 0.5, 0.6, 0.7], min_len=[0, 10, 20], interp=[0, 20], min_score=[0, 0.5], static_px=[0, 10])
PFIX = dict(static_min=60, static_score=0.6)
ABLATION = {"ninguno": [], "solo escala": ["rel_size"], "solo largo": ["min_len"], "solo interpolacion": ["interp"],
            "solo confianza": ["min_score"], "solo estatico": ["static_px"], "escala + largo": ["rel_size", "min_len"],
            "escala + largo + interp.": ["rel_size", "min_len", "interp"], "todo": list(PGRID)}


def post_seq(job):
    v, seq, tcfg, combos = job
    d = data(seq); fx = fused(v, seq)["dets"]; n = len(fx)
    A = gmc_A(seq); tr = L.run_bytetrack(fx, tcfg)
    rows = []
    for i in range(0, len(combos), 8):          # por lotes: los acumuladores de motmetrics ocupan mucha memoria en escenas densas
        cs = combos[i:i + 8]
        accs = [L.accumulate(L.tracks_to_pred(L.postprocess(tr, dict(PFIX, **c), A), d["ig"]), d["gt"], 0, n) for c in cs]
        df = L.compute(accs, [str(k) for k in range(len(cs))]); del accs
        rows += [dict(key=json.dumps(c, sort_keys=True), seq=seq, st=0, **df.iloc[k].to_dict()) for k, c in enumerate(cs)]
    print(f"   {v} {seq} listo", flush=True)
    return rows


def stage_post():
    combos = [dict(zip(PGRID, vals)) for vals in itertools.product(*PGRID.values())]
    fus = json.load(open(out_dir / "fusion_cv.json"))
    for v in (a.variants or ["YR"]):
        tcfg = fus[v]["final"]["cfg"][0]; t0 = time.time()
        print(f"[post {v}] tracker fijo {tcfg} | {len(combos)} combinaciones x {len(seqs)} secuencias", flush=True)
        with Pool(a.jobs) as pool: rows = [r for rs in pool.imap_unordered(post_seq, [(v, s, tcfg, combos) for s in seqs]) for r in rs]
        json.dump(rows, open(out_dir / f"post_rows_{v}.json", "w"))
        report = dict(variant=v, tracker=tcfg, fixed=PFIX, ablation={})
        for name, params in ABLATION.items():
            keys = [json.dumps(c, sort_keys=True) for c in combos if all(c[p] == PGRID[p][0] for p in PGRID if p not in params)]
            r = cv_select(rows, keys, f"post {v} | {name}")
            report["ablation"][name] = dict(cv_mota=r["cv_mota"], cv_idf1=r["cv_idf1"], folds=r["folds"], final=r["final"],
                                            cv_IDS=sum(f["heldout"]["IDS"] for f in r["folds"]),
                                            cv_FP=sum(f["heldout"]["FP"] for f in r["folds"]),
                                            cv_FN=sum(f["heldout"]["FN"] for f in r["folds"]))
        json.dump(report, open(out_dir / f"post_cv_{v}.json", "w"), indent=1)
        best = report["ablation"]["todo"]["final"]["cfg"]
        json.dump(dict(variant=v, final=dict(cfg=[tcfg, dict(PFIX, **json.loads(json.dumps(best)))])),
                  open(out_dir / f"post_final_{v}.json", "w"), indent=1)
        print(f"\n== {v}: ablacion (MOTA en validacion cruzada, secuencias completas) [{time.time() - t0:.0f}s] ==")
        for name, r in report["ablation"].items():
            print(f"  {name:26s} CV MOTA {r['cv_mota']:.3f}  IDF1 {r['cv_idf1']:.3f}  | 20 sec. {r['final']['in_sample']['mota']:.3f}  {r['final']['cfg']}")

# ============================================================================= etapa E2/E3: tracker + post-proceso (descartada)
POST_GRID = {
    "filter": [("min_len", [0, 3, 5, 10, 20]), ("min_score", [0, 0.3, 0.4, 0.5, 0.6])],
    "scale":  [("rel_size", [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])],
    "interp": [("interp", [0, 5, 10, 20, 40])],
    "static": [("static_px", [0, 5, 10, 20]), ("static_score", [0.5, 0.6, 0.7, 1.0]), ("static_min", [30, 60, 120])],
}
POST_GRID["all"] = POST_GRID["filter"] + POST_GRID["scale"] + POST_GRID["interp"] + POST_GRID["static"]
POST_GRID["none"] = []
TRK_GRID = [("high", [0.35, 0.45, 0.55, 0.65, 0.75]), ("low", [0.05, 0.1, 0.2, 0.3]), ("new", None),
            ("buffer", [15, 30, 60, 90]), ("match", [0.7, 0.8, 0.9])]


def stage_tune():
    """Descenso por coordenadas por pliegue. Para no repetir tracking, cada paso evalua un lote de configuraciones
    en paralelo sobre TODAS las secuencias y luego elige con las de ajuste del pliegue."""
    v = a.variant
    if v == "auto":
        rep = json.load(open(out_dir / "fusion_cv.json")); v = max(rep, key=lambda x: rep[x]["cv_mota"])
        print(f"variante elegida por la etapa fusion: {v} (CV MOTA {rep[v]['cv_mota']:.3f})", flush=True)
    label = f"tune {v} post={a.post}"; t0 = time.time()
    cache = {}                                      # key -> filas (todas las secuencias)

    def evaluate(cfgs):
        todo = [c for c in cfgs if json.dumps(c, sort_keys=True) not in cache]
        if todo:
            plan = {}
            for t, p in todo: plan.setdefault(json.dumps(t, sort_keys=True), (t, []))[1].append(p)
            plan = list(plan.values())
            for r in run_jobs([(v, s, plan) for s in seqs]): cache.setdefault(r["key"], []).append(r)

    def best_of(cfgs, subset):
        evaluate(cfgs); rows = [r for c in cfgs for r in cache[json.dumps(c, sort_keys=True)]]
        agg = aggregate(rows, subset); k = max(agg, key=lambda x: agg[x]["mota"]); return json.loads(k), agg[k]

    def descend(subset):
        hist = json.load(open(out_dir / "fusion_cv.json")).get(v, {})
        t = dict(DEFAULT_TRK, **(hist.get("final", {}).get("cfg", [{}])[0] if hist else {}))   # arranca del mejor de E1
        p = dict(DEFAULT_POST); best = None
        for name, values in TRK_GRID + POST_GRID[a.post]:
            if values is None: values = sorted({t["high"], round(t["high"] + 0.05, 2), round(t["high"] + 0.1, 2)})
            cand = []
            for val in values:
                if name in DEFAULT_TRK:
                    tt = dict(t, **{name: val})
                    if name == "high": tt["new"] = max(val, t["new"]) if t["new"] > t["high"] else val
                    if tt["low"] >= tt["high"]: continue
                    cand.append((tt, p))
                else:
                    cand.append((t, dict(p, **{name: val})))
            (t, p), best = best_of(cand, subset)
        return [t, p], best

    rep = []
    for k in range(a.folds):
        test = [s for s in seqs if fold_of[s] == k]; train = [s for s in seqs if fold_of[s] != k]
        cfg, r_tr = descend(train)
        evaluate([tuple(cfg)]); evaluate([(DEFAULT_TRK, DEFAULT_POST)])
        r_te = aggregate(cache[json.dumps(cfg, sort_keys=True)], test)[json.dumps(cfg, sort_keys=True)]
        r_def = aggregate(cache[json.dumps([DEFAULT_TRK, DEFAULT_POST], sort_keys=True)], test)
        r_def = list(r_def.values())[0]
        rep.append(dict(fold=k, test=test, cfg=cfg, train=r_tr, heldout=r_te, heldout_default=r_def))
        print(f"[{label}] pliegue {k}: ajuste={r_tr['mota']:.3f} validacion={r_te['mota']:.3f} (por defecto {r_def['mota']:.3f}) "
              f"[{time.time() - t0:.0f}s]\n      {cfg}", flush=True)
    cv = float(np.mean([r["heldout"]["mota"] for r in rep])); cv_i = float(np.mean([r["heldout"]["idf1"] for r in rep]))
    final, r_fin = descend(seqs)
    report = dict(variant=v, post=a.post, folds=rep, cv_mota=cv, cv_idf1=cv_i,
                  cv_IDS=sum(r["heldout"]["IDS"] for r in rep), cv_FP=sum(r["heldout"]["FP"] for r in rep),
                  cv_FN=sum(r["heldout"]["FN"] for r in rep), final=dict(cfg=final, in_sample=r_fin))
    json.dump(report, open(out_dir / f"tune_{v}_{a.post}.json", "w"), indent=1)
    print(f"[{label}] VALIDACION CRUZADA MOTA={cv:.3f} IDF1={cv_i:.3f} | FINAL {final} (20 sec. {r_fin['mota']:.3f}) [{time.time() - t0:.0f}s]")


# ============================================================================= corrida final (frames completos)
def final_seq(job):
    v, seq, tcfg, pcfg, out = job
    fx = fused(v, seq); n = len(fx["dets"])
    t = time.time(); tr = L.run_bytetrack(fx["dets"], tcfg); t_trk = time.time() - t
    t = time.time(); A = gmc_A(seq) if pcfg.get("static_px", 0) else None
    tr = L.postprocess(tr, pcfg, A); t_post = time.time() - t
    L.write_mot(tr, Path(out) / f"{seq}.txt")
    return seq, dict(frames=n, t_det=fx["t_det"], t_trk=t_trk + t_post, t_post=t_post, fps=n / max(fx["t_det"], 1e-9), device="cpu")


def stage_final():
    rep = json.load(open(a.cfg_from)); v = rep["variant"]; tcfg, pcfg = rep["final"]["cfg"]
    out = Path("results") / a.name; out.mkdir(parents=True, exist_ok=True)
    with Pool(a.jobs) as pool: timing = dict(pool.map(final_seq, [(v, s, tcfg, pcfg, out) for s in seqs]))
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    (out / "config.json").write_text(json.dumps(dict(variant=v, detectors=VARIANTS[v], tracker=tcfg, post=pcfg), indent=1))
    print("escrito", out, "-> evaluar con: python -I scripts/evaluate.py --names", a.name)


def _prefuse(job):
    v, s = job; t = time.time(); fused(v, s); return v, s, time.time() - t


def stage_prefuse():
    """Precalcula results/_cache/fx_<variante>/ (no necesita GT)."""
    jobs = [(v, s) for v in (a.variants or VARIANTS) for s in seqs if len(VARIANTS[v][0]) > 1]
    with Pool(a.jobs) as pool:
        for v, s, dt in pool.imap_unordered(_prefuse, jobs): print(f"fusionado {v} {s} [{dt:.0f}s]", flush=True)


{"prefuse": stage_prefuse, "fusion": stage_fusion, "post": stage_post, "tune": stage_tune, "final": stage_final}[a.stage]()
