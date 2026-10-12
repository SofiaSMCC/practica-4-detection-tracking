"""Metodo 4 FINAL (corrida end-to-end sobre imagenes, para medir FPS reales en GPU).

  frame --> YOLO26 (UAVDT, imagen completa) --+
        --> RT-DETRv4 (VisDrone)            --+--> fusion ponderada de cajas --> ByteTrack --> post-proceso --> MOT
        --> [SAHI 640 + YOLO26]             --+    (m4_lib.fuse_frame)                        (offline, al final)

La configuracion (detectores, pesos, umbral de fusion, ByteTrack y post-proceso) se lee del json de
m4_experiments.py tune (--from), asi la corrida usa exactamente lo elegido por validacion cruzada.
El post-proceso necesita el track completo (escala tipica de la escena, interpolacion), por eso es offline; su costo
se mide aparte (t_post) y es despreciable frente a la deteccion.
Salida: results/<--name>/<seq>.txt, timing.json (t_det = detectores + fusion, t_trk = ByteTrack + post), config.json
Uso (Khipu, GPU):  sbatch scripts/slurm_run.sh scripts/method4_final.py --from results/m4_exp/tune_YR_all.json [--seqs ...]
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
import m4_lib as L
from common import find_sequences, Dets, tracker_args, save_timing

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--from", dest="cfg_from", required=True, help="results/m4_exp/tune_<variante>_<post>.json")
ap.add_argument("--root", default="data"); ap.add_argument("--seqs", nargs="*")
ap.add_argument("--name", default="method4_sahi_fusion_bytetrack_gpu")
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--max_frames", type=int)
a = ap.parse_args()

rep = json.loads(Path(a.cfg_from).read_text()); variant = rep["variant"]; tcfg, pcfg = rep["final"]["cfg"]
VARIANTS = L.VARIANTS
dets_names, weights, thr, *slots = VARIANTS[variant]; slots = slots[0] if slots else None
print(f"variante {variant}: {dets_names} pesos={weights} thr={thr} slots={slots}\ntracker={tcfg}\npost={pcfg}", flush=True)

# ----------------------------------------------------------------------------- detectores (mismos ajustes que cache_detections.py)
dev = a.device
DETECT = {}
if "yolo26" in dets_names or "sahi" in dets_names:
    from ultralytics import YOLO
    yolo = YOLO("weights/uavdt_yolo26m.pt")
    def det_yolo(img):
        b = yolo.predict(img, conf=0.05, imgsz=1024, device=dev, verbose=False, nms=False)[0].boxes.cpu().numpy()
        return np.column_stack([b.xyxy, b.conf]) if len(b) else np.zeros((0, 5))
    DETECT["yolo26"] = det_yolo
if "rtdetr" in dets_names:
    from rtdetr_det import load_detector
    rt = load_detector("m", dev)
    def det_rt(img):
        box, scr, lab = rt(img, 0.05); k = np.isin(lab, list(L.VISDRONE_VEH))
        return np.column_stack([box[k], scr[k]]) if k.any() else np.zeros((0, 5))
    DETECT["rtdetr"] = det_rt
if "sahi" in dets_names:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    sm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path="weights/uavdt_yolo26m.pt",
                                            confidence_threshold=0.05, image_size=1024, device=dev)
    def det_sahi(img):
        r = get_sliced_prediction(img[:, :, ::-1], sm, slice_height=640, slice_width=640, overlap_height_ratio=0.2,
                                  overlap_width_ratio=0.2, perform_standard_pred=True, postprocess_type="GREEDYNMM",
                                  postprocess_match_metric="IOS", postprocess_match_threshold=0.5, verbose=0)
        return np.array([[o.bbox.minx, o.bbox.miny, o.bbox.maxx, o.bbox.maxy, o.score.value] for o in r.object_prediction_list]).reshape(-1, 5)
    DETECT["sahi"] = det_sahi


def sync():
    if dev.startswith("cuda"): torch.cuda.synchronize()


out = Path("results") / a.name; out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps(dict(variant=variant, detectors=VARIANTS[variant], tracker=tcfg, post=pcfg), indent=1))
from ultralytics.trackers.byte_tracker import BYTETracker
for seq, d in find_sequences(a.root, a.seqs):
    imgs = sorted(d.glob("img*.jpg"))[: a.max_frames]
    for p in imgs[:3]: [DETECT[n](cv2.imread(str(p))) for n in dets_names]           # calentamiento (no se mide)
    tr = BYTETracker(tracker_args("bytetrack.yaml", track_high_thresh=tcfg["high"], track_low_thresh=tcfg["low"],
                                  new_track_thresh=tcfg["new"], track_buffer=tcfg["buffer"], match_thresh=tcfg["match"]))
    tracks, t_det, t_trk = {}, 0.0, 0.0; per_det = {n: 0.0 for n in dets_names}
    for k, p in enumerate(imgs):
        img = cv2.imread(str(p))
        sync(); t0 = time.time(); per = []
        for n in dets_names:
            t1 = time.time(); x = DETECT[n](img).astype(np.float32); sync(); per_det[n] += time.time() - t1; per.append(x)
        fx = per[0] if len(per) == 1 else L.fuse_frame(per, weights, thr, slots)
        t_det += time.time() - t0
        t0 = time.time()
        fx = fx[fx[:, 4] >= tcfg["low"]] if len(fx) else fx
        for r in tr.update(Dets(fx[:, :4], fx[:, 4], np.ones(len(fx))), L.DUMMY):
            tracks.setdefault(int(r[4]), {})[k + 1] = (float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[5]))
        t_trk += time.time() - t0
        if k % 200 == 0: print(f"{seq} {k}/{len(imgs)}  {(k + 1) / max(t_det + t_trk, 1e-9):.1f} FPS", flush=True)
    t0 = time.time()
    A = None
    if pcfg.get("static_px", 0):
        from ultralytics.trackers.utils.gmc import GMC
        g = GMC(method="sparseOptFlow"); A = L.cum_affine([np.array(g.apply(cv2.imread(str(p)), None))[:2] for p in imgs])
    tracks = L.postprocess(tracks, pcfg, A); t_post = time.time() - t0
    L.write_mot(tracks, out / f"{seq}.txt")
    n = len(imgs)
    save_timing(out, {seq: dict(frames=n, t_det=t_det, t_trk=t_trk + t_post, t_post=t_post, fps=n / (t_det + t_trk + t_post),
                                det_ms={k: 1e3 * v / n for k, v in per_det.items()}, device=dev)})
    print(f"{seq}: {n / (t_det + t_trk + t_post):.1f} FPS  det {1e3 * t_det / n:.1f} ms  trk {1e3 * t_trk / n:.1f} ms  post {t_post:.1f}s", flush=True)
print("listo ->", out, "| evaluar: python -I scripts/evaluate.py --names", a.name)
