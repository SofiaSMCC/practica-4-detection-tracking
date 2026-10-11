"""Metodo 2 (eficiencia): YOLO26 (pesos UAVDT) + ByteTrack.

Dos ideas nuevas respecto al metodo 1:
  * YOLO26 puede detectar SIN NMS ("NMS-free"): la red ya entrega una caja por objeto, asi que se ahorra el paso
    que elimina cajas duplicadas. Con --with_nms se activa el NMS clasico para comparar la latencia.
  * ByteTrack no tira las detecciones de baja confianza: las usa en una SEGUNDA ronda de emparejamiento para
    recuperar vehiculos tapados u opacos (que el detector ve "con duda") en vez de perderlos.

Pipeline por frame:   imagen --YOLO26--> detecciones (conf >= track_low) --ByteTrack--> tracks con ID --> linea MOT
Salida MOT: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<vis>   (class: 1=car 2=truck 3=bus)

Uso: python -I scripts/method2_yolo26_bytetrack.py [--seqs M0208] [--max_frames 50] [--device cpu] [--with_nms]
"""
import argparse, json, sys, time
from pathlib import Path
import cv2
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

sys.path.insert(0, str(Path(__file__).parent))
from common import find_sequences, Dets, tracker_args, tracks_to_mot, validate_mot, save_timing

# ----------------------------------------------------------------------------- 0. argumentos
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default="data")
ap.add_argument("--seqs", nargs="*", help="secuencias (por defecto las 20 de test)")
ap.add_argument("--name", help="carpeta de salida en results/ (por defecto segun NMS)")
ap.add_argument("--weights", default="weights/uavdt_yolo26m.pt")
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--device", default=None)
ap.add_argument("--with_nms", action="store_true", help="usar NMS clasico en vez de la cabeza NMS-free (para medir la latencia)")
ap.add_argument("--track_high", type=float, help="ByteTrack: conf minima de la 1a ronda y para abrir tracks (def. 0.25)")
ap.add_argument("--track_low", type=float, help="ByteTrack: conf minima de la 2a ronda (def. 0.1)")
ap.add_argument("--new_track", type=float, help="ByteTrack: conf minima para crear un track nuevo (def. 0.25)")
ap.add_argument("--buffer", type=int, help="ByteTrack: frames que sobrevive un track perdido (def. 30)")
ap.add_argument("--match", type=float, help="ByteTrack: umbral de distancia para emparejar, mas bajo = mas estricto (def. 0.8)")
ap.add_argument("--max_frames", type=int, help="solo los primeros N frames (pruebas rapidas)")
a = ap.parse_args()

# ----------------------------------------------------------------------------- 1. detector + configuracion del tracker
cfg = tracker_args("bytetrack.yaml", track_high_thresh=a.track_high, track_low_thresh=a.track_low,
                   new_track_thresh=a.new_track, track_buffer=a.buffer, match_thresh=a.match)
model = YOLO(a.weights)   # clases del modelo: 0=car, 1=truck, 2=bus
# El detector debe dejar pasar tambien las detecciones "dudosas": ByteTrack las necesita para su 2a ronda.
det_conf = cfg.track_low_thresh

out_dir = Path("results") / (a.name or ("method2_yolo26_nms_bytetrack" if a.with_nms else "method2_yolo26_bytetrack"))
out_dir.mkdir(parents=True, exist_ok=True)
timing = {}

for seq, img_dir in find_sequences(a.root, a.seqs):
    # ------------------------------------------------------------------------- 2. un tracker NUEVO por secuencia
    tracker = BYTETracker(cfg)
    images = sorted(img_dir.glob("img*.jpg"))[: a.max_frames]
    lines, t_det, t_trk = [], 0.0, 0.0

    for frame, path in enumerate(images, start=1):
        img = cv2.imread(str(path))

        # --------------------------------------------------------------------- 3. DETECCION (solo el modelo)
        # nms=False -> cabeza NMS-free de YOLO26 ; nms=None -> NMS externo clasico
        t0 = time.perf_counter()
        b = model.predict(img, conf=det_conf, imgsz=a.imgsz, device=a.device, nms=None if a.with_nms else False,
                          verbose=False)[0].boxes.cpu().numpy()
        t1 = time.perf_counter()

        # --------------------------------------------------------------------- 4. ASOCIACION (solo ByteTrack)
        dets = Dets(b.xyxy, b.conf, b.cls + 1)            # clase del modelo 0/1/2 -> UAVDT 1/2/3
        out = tracker.update(dets, img)                   # filas: [x1,y1,x2,y2, id, conf, clase, idx_deteccion]
        t2 = time.perf_counter()
        t_det += t1 - t0
        t_trk += t2 - t1

        # --------------------------------------------------------------------- 5. escribir en formato MOT
        lines += tracks_to_mot(frame, out)

    (out_dir / f"{seq}.txt").write_text("\n".join(lines) + "\n")
    timing[seq] = dict(frames=len(images), t_det=t_det, t_trk=t_trk, fps=len(images) / (t_det + t_trk), device=str(a.device or "auto"))
    print(seq, timing[seq], flush=True)
    save_timing(out_dir, timing)   # tras CADA secuencia: si el job se corta, no se pierde lo hecho
    # ------------------------------------------------------------------------- 6. validar el formato MOT
    validate_mot(seq, lines)

