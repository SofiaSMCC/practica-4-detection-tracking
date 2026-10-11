"""Guarda las detecciones CRUDAS (conf >= 0.05) de un detector sobre las 20 secuencias de test, para poder ajustar
los hiperparametros del tracker sin volver a detectar. Un archivo por secuencia: <out>/<detector>/<seq>.pkl con
  dets : lista (un elemento por frame) de arrays (N,6) = [x1, y1, x2, y2, conf, clase_cruda]
  t_det: tiempo total de deteccion (s)   frames: numero de frames
Detectores: yolo11 (UAVDT), yolo26 (UAVDT, sin NMS), rtdetr (VisDrone; etiquetas VisDrone), sahi (YOLO26 + SAHI 640/0.2),
            gmc (no es un detector: transformacion afin de camara por frame, sparseOptFlow, para BoT-SORT).
Si el archivo ya existe se salta (varios procesos pueden repartirse el trabajo; --reverse recorre las secuencias al reves).
Uso: python -I scripts/cache_detections.py --detector yolo26 --device cpu --threads 8 [--seqs M0208] [--max_frames N]"""
import argparse, os, pickle, sys, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import find_sequences

ap = argparse.ArgumentParser()
ap.add_argument("--detector", required=True, choices=["yolo11", "yolo26", "rtdetr", "sahi", "gmc"])
ap.add_argument("--root", default="data"); ap.add_argument("--out", default="results/_cache")
ap.add_argument("--seqs", nargs="*"); ap.add_argument("--max_frames", type=int)
ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--conf", type=float, default=0.05); ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--slice", type=int, default=640); ap.add_argument("--overlap", type=float, default=0.2)
ap.add_argument("--reverse", action="store_true")
ap.add_argument("--tag", help="subcarpeta de salida (por defecto el nombre del detector); util para variantes, p.ej. sahi320")
a = ap.parse_args()
import torch
torch.set_num_threads(a.threads)

if a.detector in ("yolo11", "yolo26"):
    from ultralytics import YOLO
    model = YOLO(f"weights/uavdt_{a.detector}m.pt")
    def detect(img):
        b = model.predict(img, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False,
                          **({"nms": False} if a.detector == "yolo26" else {}))[0].boxes.cpu().numpy()
        return np.column_stack([b.xyxy, b.conf, b.cls]) if len(b) else np.zeros((0, 6))
elif a.detector == "rtdetr":
    from rtdetr_det import load_detector
    det = load_detector("m", a.device)
    def detect(img):
        box, scr, lab = det(img, a.conf)
        return np.column_stack([box, scr, lab]) if len(box) else np.zeros((0, 6))
elif a.detector == "sahi":
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    m = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path="weights/uavdt_yolo26m.pt",
                                           confidence_threshold=a.conf, image_size=a.imgsz, device=a.device)
    def detect(img):
        r = get_sliced_prediction(img[:, :, ::-1], m, slice_height=a.slice, slice_width=a.slice,
                                  overlap_height_ratio=a.overlap, overlap_width_ratio=a.overlap, perform_standard_pred=True,
                                  postprocess_type="GREEDYNMM", postprocess_match_metric="IOS", postprocess_match_threshold=0.5, verbose=0)
        p = r.object_prediction_list
        return np.array([[o.bbox.minx, o.bbox.miny, o.bbox.maxx, o.bbox.maxy, o.score.value, o.category.id] for o in p]).reshape(-1, 6)
else:
    from ultralytics.trackers.utils.gmc import GMC

out = Path(a.out) / (a.tag or a.detector); out.mkdir(parents=True, exist_ok=True)
seqs = find_sequences(a.root, a.seqs)
if a.reverse: seqs = seqs[::-1]
for seq, d in seqs:
    f_out = out / f"{seq}.pkl"
    if f_out.exists(): print("ya existe:", seq, flush=True); continue
    imgs = sorted(d.glob("img*.jpg"))[: a.max_frames]; t0 = time.time(); res = []
    gmc = GMC(method="sparseOptFlow") if a.detector == "gmc" else None
    for k, p in enumerate(imgs):
        img = cv2.imread(str(p))
        res.append(np.array(gmc.apply(img, None), dtype=np.float64)[:2] if gmc else detect(img))
        if k % 100 == 0: print(f"{a.detector} {seq} {k}/{len(imgs)} ({time.time() - t0:.0f}s)", flush=True)
    tmp = f_out.with_suffix(".tmp")
    pickle.dump({"warps" if gmc else "dets": res, "t_det": time.time() - t0, "frames": len(imgs)}, open(tmp, "wb"))
    os.replace(tmp, f_out); print("guardado", f_out, f"({time.time() - t0:.0f}s)", flush=True)
