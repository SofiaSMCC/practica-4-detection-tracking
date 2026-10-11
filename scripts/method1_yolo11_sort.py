"""Metodo 1 (baseline): YOLOv11 (pesos UAVDT) + SORT. Asociacion puramente geometrica.

Pipeline por frame:   imagen --YOLO--> detecciones --SORT--> tracks con ID --> linea MOT

Salida (formato MOT Challenge), una linea por vehiculo y frame, en results/<name>/<seq>.txt:
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<vis>
      frame : indice del frame, desde 1          id   : identificador persistente del track
      conf  : confianza del detector             class: 1=car, 2=truck, 3=bus (convencion UAVDT)
      vis   : visibilidad, fijada a 1 en predicciones

Uso: python -I scripts/method1_yolo11_sort.py [--seqs M0208 M1004] [--max_frames 50] [--device cpu]
"""
import argparse, json, sys, time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import find_sequences, tracks_to_mot, validate_mot, save_timing
from sort import Sort

# ----------------------------------------------------------------------------- 0. argumentos
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default="data", help="carpeta con UAV-benchmark-M/ y M_attr/")
ap.add_argument("--seqs", nargs="*", help="secuencias a correr (por defecto las 20 de test)")
ap.add_argument("--name", default="method1_yolo11_sort", help="carpeta de salida en results/")
ap.add_argument("--weights", default="weights/uavdt_yolo11m.pt")
ap.add_argument("--conf", type=float, default=0.25, help="umbral de confianza del detector")
ap.add_argument("--imgsz", type=int, default=1024, help="tamano de entrada del detector")
ap.add_argument("--device", default=None, help="'cpu', '0', ... (por defecto: GPU si hay)")
ap.add_argument("--max_age", type=int, default=30, help="SORT: frames sin deteccion antes de borrar un track")
ap.add_argument("--min_hits", type=int, default=3, help="SORT: detecciones para confirmar un track")
ap.add_argument("--iou", type=float, default=0.3, help="SORT: IoU minimo para emparejar")
ap.add_argument("--max_frames", type=int, help="solo los primeros N frames de cada secuencia (pruebas rapidas)")
a = ap.parse_args()

# ----------------------------------------------------------------------------- 1. detector (pesos preentrenados)
model = YOLO(a.weights)   # clases del modelo: 0=car, 1=truck, 2=bus -> ya son solo vehiculos

out_dir = Path("results") / a.name
out_dir.mkdir(parents=True, exist_ok=True)
timing = {}

for seq, img_dir in find_sequences(a.root, a.seqs):
    # ------------------------------------------------------------------------- 2. un tracker NUEVO por secuencia
    tracker = Sort(max_age=a.max_age, min_hits=a.min_hits, iou_thr=a.iou)
    images = sorted(img_dir.glob("img*.jpg"))[: a.max_frames]
    lines, t_det, t_trk = [], 0.0, 0.0

    for frame, path in enumerate(images, start=1):        # MOT numera los frames desde 1
        img = cv2.imread(str(path))

        # --------------------------------------------------------------------- 3. DETECCION (solo el modelo)
        t0 = time.perf_counter()
        boxes = model.predict(img, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0].boxes.cpu().numpy()
        t1 = time.perf_counter()

        # --------------------------------------------------------------------- 4. ASOCIACION (solo SORT)
        # dets: [x1, y1, x2, y2, conf, clase]; clase 0/1/2 del modelo -> 1/2/3 de UAVDT
        dets = np.column_stack([boxes.xyxy, boxes.conf, boxes.cls + 1]) if len(boxes) else np.zeros((0, 6))
        tracks = tracker.update(dets)                     # (M,7): [x1, y1, x2, y2, id, conf, clase]
        t2 = time.perf_counter()
        t_det += t1 - t0
        t_trk += t2 - t1

        # --------------------------------------------------------------------- 5. escribir en formato MOT
        lines += tracks_to_mot(frame, tracks)

    (out_dir / f"{seq}.txt").write_text("\n".join(lines) + "\n")
    timing[seq] = dict(frames=len(images), t_det=t_det, t_trk=t_trk, fps=len(images) / (t_det + t_trk), device=str(a.device or "auto"))
    print(seq, timing[seq], flush=True)
    save_timing(out_dir, timing)   # tras CADA secuencia: si el job se corta, no se pierde lo hecho

    # ------------------------------------------------------------------------- 6. validar el formato MOT
    validate_mot(seq, lines)

