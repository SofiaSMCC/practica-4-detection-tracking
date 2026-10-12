"""Metodo 4 (version final): piezas reutilizables, todas sobre detecciones CACHEADAS (results/_cache/<detector>/<seq>.pkl).

  1. fuse_frame      fusion ponderada de cajas de varios detectores (tipo Weighted Boxes Fusion, Solovyev et al. 2021)
  2. run_bytetrack   ByteTrack sobre las detecciones fusionadas; devuelve filas con la CONFIANZA del track (para el post-proceso)
  3. postprocess     heuristicas offline sobre los tracks: (a) borrar tracks cortos / de baja confianza,
                     (b) borrar tracks estaticos con movimiento de camara compensado (GMC cacheado),
                     (c) interpolar huecos cortos dentro de un track (idea de GSI/StrongSORT, version lineal)
  4. score_seq       motmetrics con el mismo protocolo que evaluate.py (IoU >= 0.5, regiones ignoradas del GT)

Formato interno de un track: dict  tid -> {frame: (x1, y1, x2, y2, score)}   (frames 1-based)
"""
import pickle
from pathlib import Path
import numpy as np

# variantes de fusion (E1)
# nombre: (detectores, pesos, umbral IoU de fusion).  Y = YOLO26 (UAVDT), R = RT-DETRv4 (VisDrone), S = SAHI 640 + YOLO26
# YOLO26 y SAHI usan los MISMOS pesos: sus errores estan correlacionados, por eso se prueban pesos que los cuentan como
# 'medio voto' cada uno (Y0.5 R1 S0.5) para que su acuerdo no valga como consenso entre modelos distintos.
VARIANTS = {
    "S":        (["sahi"], [1], 0.55),                                   # metodo 4 anterior (SAHI solo)
    "Y":        (["yolo26"], [1], 0.55),                                 # metodo 2 (referencia)
    "YR":       (["yolo26", "rtdetr"], [1, 1], 0.55),                    # metodo 5 anterior (fusion sin SAHI)
    "RS":       (["rtdetr", "sahi"], [1, 1], 0.55),                      # SAHI reemplaza a YOLO26 en la fusion
    "YRS":      (["yolo26", "rtdetr", "sahi"], [1, 1, 1], 0.55),
    "YRS_half": (["yolo26", "rtdetr", "sahi"], [0.5, 1, 0.5], 0.55),
    "YRS_r15":  (["yolo26", "rtdetr", "sahi"], [1, 1.5, 1], 0.55),
    "YRS_t50":  (["yolo26", "rtdetr", "sahi"], [1, 1, 1], 0.50),
    "YRS_t65":  (["yolo26", "rtdetr", "sahi"], [1, 1, 1], 0.65),
    # SAHI como RESPALDO de YOLO26: comparten un slot (un solo voto); SAHI entra solo donde YOLO26 no vio nada
    "YRS_slot":     (["yolo26", "rtdetr", "sahi"], [1, 1, 1], 0.55, [0, 1, 0]),
    "YRS_slot_r15": (["yolo26", "rtdetr", "sahi"], [1, 1.5, 1], 0.55, [0, 1, 0]),
    "YRS_slot_t50": (["yolo26", "rtdetr", "sahi"], [1, 1, 1], 0.50, [0, 1, 0]),
}
VISDRONE_VEH = {3, 4, 5, 8}          # car, van, truck, bus en RT-DETR (VisDrone)
DUMMY = np.zeros((2, 2, 3), np.uint8)


# ----------------------------------------------------------------------------- carga
def load_cache(cache_dir, det, seq, min_conf=0.05):
    """Detecciones (N,5) [x1,y1,x2,y2,conf] por frame; RT-DETR se filtra a clases de vehiculo."""
    d = pickle.load(open(Path(cache_dir) / det / f"{seq}.pkl", "rb"))
    out = []
    for f in d["dets"]:
        if len(f):
            keep = f[:, 4] >= min_conf
            if det.startswith("rtdetr"): keep &= np.isin(f[:, 5].astype(int), list(VISDRONE_VEH))
            f = f[keep][:, :5]
        out.append(f.astype(np.float32) if len(f) else np.zeros((0, 5), np.float32))
    return out, d["t_det"]


# ----------------------------------------------------------------------------- 1. fusion
def _iou_many(b, B):
    x1 = np.maximum(b[0], B[:, 0]); y1 = np.maximum(b[1], B[:, 1])
    x2 = np.minimum(b[2], B[:, 2]); y2 = np.minimum(b[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def fuse_frame(per_model, weights, thr=0.55, slots=None):
    """per_model: lista de (N_m,5). weights: peso de cada modelo. slots: 'voto' de cada modelo (por defecto uno por modelo).
    Dos modelos con el mismo slot (p.ej. YOLO26 y SAHI+YOLO26, mismos pesos) cuentan como UN voto: un grupo acepta solo
    a uno de ellos (el de mayor confianza), asi SAHI solo aporta cuando la pasada a imagen completa no vio el objeto.
    Agrupa voraz de mayor a menor (conf * peso): una caja entra al grupo cuya caja fusionada tenga IoU >= thr y que aun
    no tenga un miembro de ese modelo. Caja fusionada = promedio ponderado por conf*peso.
    Score = sum(peso_m * conf_m) / sum(pesos)  -> una caja vista por un solo modelo queda penalizada (hace falta consenso)."""
    slots = list(range(len(per_model))) if slots is None else slots
    W = float(sum({sl: w for sl, w in zip(slots, weights)}.values()))   # un peso por slot
    rows = [(w * r[4], slots[m], r) for m, (f, w) in enumerate(zip(per_model, weights)) for r in f]
    if not rows: return np.zeros((0, 5), np.float32)
    rows.sort(key=lambda t: -t[0])
    boxes = np.zeros((len(rows), 4)); wsum = np.zeros(len(rows)); ssum = np.zeros(len(rows)); acc = np.zeros((len(rows), 4))
    members = []; n = 0
    shared = {sl for sl in slots if slots.count(sl) > 1}
    for wc, m, r in rows:
        gi = -1; blocked = False
        if n:
            iou = _iou_many(r[:4], boxes[:n])
            for j in np.argsort(-iou):
                if iou[j] < thr: break
                if m not in members[j]: gi = j; break
                blocked = True
        if gi < 0 and blocked and m in shared:
            continue          # su slot ya voto por este objeto (p.ej. YOLO26 ya lo vio): SAHI no crea un duplicado
        if gi < 0:
            gi = n; n += 1; members.append(set())
        members[gi].add(m); acc[gi] += wc * r[:4]; wsum[gi] += wc; ssum[gi] += wc
        boxes[gi] = acc[gi] / wsum[gi]
    return np.column_stack([boxes[:n], ssum[:n] / W]).astype(np.float32)


def fuse_sequence(dets_by_model, weights, thr, slots=None):
    nfr = min(len(d) for d in dets_by_model)
    return [fuse_frame([d[k] for d in dets_by_model], weights, thr, slots) for k in range(nfr)]


# ----------------------------------------------------------------------------- 2. tracking
def run_bytetrack(frames, cfg, st=0, en=None):
    """ByteTrack (ultralytics) sobre frames[st:en]. cfg: high, low, new, buffer, match. -> tracks dict."""
    from common import Dets, tracker_args
    from ultralytics.trackers.byte_tracker import BYTETracker
    tr = BYTETracker(tracker_args("bytetrack.yaml", track_high_thresh=cfg["high"], track_low_thresh=cfg["low"],
                                  new_track_thresh=cfg["new"], track_buffer=cfg["buffer"], match_thresh=cfg["match"]))
    en = len(frames) if en is None else en
    tracks = {}
    for k in range(st, en):
        d = frames[k]
        d = d[d[:, 4] >= cfg["low"]] if len(d) else d
        out = tr.update(Dets(d[:, :4], d[:, 4], np.ones(len(d))), DUMMY)
        for r in out:
            tracks.setdefault(int(r[4]), {})[k + 1] = (float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[5]))
    return tracks


# ----------------------------------------------------------------------------- 3. post-proceso
def cum_affine(warps):
    """warps[k]: afin 2x3 del frame k-1 al k (GMC de BoT-SORT). -> A[k] (3x3): coordenadas del frame 1 -> frame k+1."""
    A = np.zeros((len(warps), 3, 3)); cur = np.eye(3)
    for k, w in enumerate(warps):
        H = np.vstack([np.asarray(w, float).reshape(2, 3), [0, 0, 1]])
        cur = H @ cur if k else np.eye(3); A[k] = cur
    return A


def postprocess(tracks, cfg, A=None):
    """cfg (todas opcionales, 0 = apagado):
       min_len   borra tracks con menos de min_len frames observados
       min_score borra tracks con confianza media < min_score
       static_px borra tracks 'estaticos': desplazamiento total compensado < static_px px y vida >= static_min frames,
                 solo si su confianza media < static_score (los autos detenidos con alta confianza se quedan)
       rel_size  borra tracks cuyo lado mediano sqrt(w*h) < rel_size * (lado tipico de la escena). El lado tipico es la
                 mediana de los lados medianos de los tracks de >= 10 frames (sin usar GT). Idea: en video de dron la
                 altura es casi constante, asi que los vehiculos de una escena tienen escala parecida y las cajas muy
                 pequenas suelen ser ruido (luces, objetos lejanos sin anotar, fragmentos)
       interp    rellena huecos de hasta `interp` frames con interpolacion lineal (sin extrapolar)"""
    out = {}
    side = {t: np.median([np.sqrt(max(o[2] - o[0], 0) * max(o[3] - o[1], 0)) for o in obs.values()]) for t, obs in tracks.items()}
    rs = cfg.get("rel_size", 0)
    ref = np.median([side[t] for t, obs in tracks.items() if len(obs) >= 10] or [0]) if rs else 0
    for tid, obs in tracks.items():
        fr = sorted(obs)
        if len(fr) < cfg.get("min_len", 0): continue
        if rs and side[tid] < rs * ref: continue
        sc = np.mean([obs[f][4] for f in fr])
        if sc < cfg.get("min_score", 0): continue
        if cfg.get("static_px", 0) and A is not None and len(fr) >= cfg.get("static_min", 30) and sc < cfg.get("static_score", 1.0):
            (a1, b1, c1, d1, _), (a2, b2, c2, d2, _) = obs[fr[0]], obs[fr[-1]]
            p0 = np.array([(a1 + c1) / 2, (b1 + d1) / 2, 1.0]); p1 = np.array([(a2 + c2) / 2, (b2 + d2) / 2])
            T = A[fr[-1] - 1] @ np.linalg.inv(A[fr[0] - 1])        # lleva el centro inicial al sistema del ultimo frame
            if np.linalg.norm((T @ p0)[:2] - p1) < cfg["static_px"]: continue
        new = dict(obs)
        G = cfg.get("interp", 0)
        if G:
            for f0, f1 in zip(fr[:-1], fr[1:]):
                gap = f1 - f0 - 1
                if 0 < gap <= G:
                    b0, b1 = np.array(obs[f0]), np.array(obs[f1])
                    for j in range(1, gap + 1):
                        new[f0 + j] = tuple(b0 + (b1 - b0) * j / (gap + 1))
        out[tid] = new
    return out


# ----------------------------------------------------------------------------- 4. evaluacion
def inside_ignore(b, igs_f):
    return any(b[0] > i[0] and b[1] > i[1] and b[0] + b[2] < i[0] + i[2] and b[1] + b[3] < i[1] + i[3] for i in igs_f)


def tracks_to_pred(tracks, igs=None):
    """-> {frame: [(tid, l, t, w, h)]}, descartando cajas dentro de regiones ignoradas (como evaluate.py)."""
    pred = {}
    for tid, obs in tracks.items():
        for f, (x1, y1, x2, y2, _) in obs.items():
            b = (x1, y1, x2 - x1, y2 - y1)
            if igs and inside_ignore(b, igs.get(f, [])): continue
            pred.setdefault(f, []).append((tid, *b))
    return pred


def accumulate(pred, gt, st, en):
    import motmetrics as mm
    from eval_utils import iou_dist
    acc = mm.MOTAccumulator(auto_id=True)
    for f in range(st + 1, en + 1):
        g, p = gt.get(f, []), pred.get(f, [])
        dist = iou_dist([x[1:] for x in g], [x[1:] for x in p]) if g and p else []
        acc.update([x[0] for x in g], [x[0] for x in p], dist)
    return acc


METRICS = ["mota", "idf1", "num_switches", "num_false_positives", "num_misses"]


def compute(accs, names):
    import motmetrics as mm
    return mm.metrics.create().compute_many(accs, metrics=METRICS, names=names, generate_overall=False)


def write_mot(tracks, path):
    lines = [f"{f},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f},1,1"
             for tid, obs in tracks.items() for f, (x1, y1, x2, y2, s) in obs.items()]
    lines.sort(key=lambda l: (int(l.split(",")[0]), int(l.split(",")[1])))
    Path(path).write_text("\n".join(lines) + "\n")
