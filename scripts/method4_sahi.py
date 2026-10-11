"""Metodo 4 (resolucion): SAHI (Slicing Aided Hyper Inference) + un tracker.

Problema: en fotos de dron, un auto puede medir 15x15 px dentro de una imagen de 1024 px. Al reducir la imagen
para la red, esos autos casi desaparecen y el detector no los ve.
SAHI: en vez de pasar la imagen entera, la CORTA en recuadros (slices) con algo de solape, detecta en cada uno
(alli los autos se ven mas grandes), y luego junta todo: vuelve a coordenadas de la imagen original y fusiona las
cajas duplicadas de los bordes. Cuesta mas computo (varias inferencias por frame), pero detecta mas vehiculos pequenos.

Pipeline por frame:   imagen --SAHI(YOLO por slices)--> detecciones --ByteTrack o SORT--> tracks con ID --> linea MOT
Salida MOT: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<vis>   (class: 1=car 2=truck 3=bus)

Heuristicas ajustables (tu "innovacion"): --slice (tamano del recuadro), --overlap (solape), --nms_thr (fusion de duplicados),
--no_full (no incluir la pasada sobre la imagen completa), --det/--tracker (que detector y que tracker).
Uso: python -I scripts/method4_sahi.py [--det yolo26|yolo11] [--tracker bytetrack|sort] [--seqs M0208] [--device cpu]
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

sys.path.insert(0, str(Path(__file__).parent))
from common import find_sequences, Dets, tracker_args, tracks_to_mot, validate_mot, save_timing
from sort import Sort
from ultralytics.trackers.byte_tracker import BYTETracker

# ----------------------------------------------------------------------------- 0. argumentos
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default="data")
ap.add_argument("--seqs", nargs="*", help="secuencias (por defecto las 20 de test)")
ap.add_argument("--name", help="carpeta de salida en results/")
ap.add_argument("--det", default="yolo26", choices=["yolo26", "yolo11"], help="detector base")
ap.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "sort"])
ap.add_argument("--conf", type=float, default=0.25, help="conf minima (solo SORT; ByteTrack usa 0.1 y decide el mismo)")
ap.add_argument("--imgsz", type=int, default=1024, help="tamano de entrada de la red para cada slice")
ap.add_argument("--slice", type=int, default=640, help="lado del recuadro en pixeles")
ap.add_argument("--overlap", type=float, default=0.2, help="solape entre recuadros (0-1)")
ap.add_argument("--nms_thr", type=float, default=0.5, help="umbral para fusionar cajas duplicadas entre slices")
ap.add_argument("--no_full", action="store_true", help="NO hacer ademas la deteccion sobre la imagen completa")
ap.add_argument("--device", default="cuda:0", help="'cpu' o 'cuda:0'")
ap.add_argument("--max_frames", type=int, help="solo los primeros N frames (pruebas rapidas)")
a = ap.parse_args()

# ----------------------------------------------------------------------------- 1. detector (envuelto por SAHI) y tracker
cfg = tracker_args("bytetrack.yaml")
det_conf = cfg.track_low_thresh if a.tracker == "bytetrack" else a.conf   # ByteTrack necesita las detecciones dudosas
sahi_model = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=f"weights/uavdt_{a.det}m.pt",
                                                confidence_threshold=det_conf, image_size=a.imgsz, device=a.device)

def detect(img_bgr):
    """SAHI: corta, detecta en cada recuadro, fusiona. Devuelve Dets con clases UAVDT (1/2/3)."""
    r = get_sliced_prediction(img_bgr[:, :, ::-1], sahi_model,           # SAHI espera RGB, OpenCV da BGR
                              slice_height=a.slice, slice_width=a.slice,
                              overlap_height_ratio=a.overlap, overlap_width_ratio=a.overlap,
                              perform_standard_pred=not a.no_full,        # extra: tambien la imagen entera (autos grandes)
                              postprocess_type="GREEDYNMM", postprocess_match_metric="IOS",
                              postprocess_match_threshold=a.nms_thr, verbose=0)
    p = r.object_prediction_list
    return Dets([[o.bbox.minx, o.bbox.miny, o.bbox.maxx, o.bbox.maxy] for o in p],
                [o.score.value for o in p], [o.category.id + 1 for o in p])   # clase del modelo 0/1/2 -> 1/2/3

class SortAdapter:
    """Hace que SORT (matriz numpy) hable el mismo idioma que ByteTrack (objeto Dets)."""
    def __init__(self): self.t = Sort()
    def update(self, d, img): return self.t.update(np.column_stack([d.xyxy, d.conf, d.cls]))

def make_tracker():
    return BYTETracker(cfg) if a.tracker == "bytetrack" else SortAdapter()

out_dir = Path("results") / (a.name or f"method4_sahi_{a.det}_{a.tracker}")
out_dir.mkdir(parents=True, exist_ok=True)
timing = {}

for seq, img_dir in find_sequences(a.root, a.seqs):
    # ------------------------------------------------------------------------- 2. un tracker NUEVO por secuencia
    tracker = make_tracker()
    images = sorted(img_dir.glob("img*.jpg"))[: a.max_frames]
    lines, t_det, t_trk = [], 0.0, 0.0

    for frame, path in enumerate(images, start=1):
        img = cv2.imread(str(path))

        # --------------------------------------------------------------------- 3. DETECCION por slices (SAHI)
        t0 = time.perf_counter()
        dets = detect(img)
        t1 = time.perf_counter()

        # --------------------------------------------------------------------- 4. ASOCIACION
        out = tracker.update(dets, img)                   # filas: [x1,y1,x2,y2, id, conf, clase, ...]
        t2 = time.perf_counter()
        t_det += t1 - t0
        t_trk += t2 - t1

        # --------------------------------------------------------------------- 5. escribir en formato MOT
        lines += tracks_to_mot(frame, out)

    (out_dir / f"{seq}.txt").write_text("\n".join(lines) + "\n")
    timing[seq] = dict(frames=len(images), t_det=t_det, t_trk=t_trk, fps=len(images) / (t_det + t_trk), device=str(a.device))
    print(seq, timing[seq], flush=True)
    save_timing(out_dir, timing)   # tras CADA secuencia: si el job se corta, no se pierde lo hecho
    # ------------------------------------------------------------------------- 6. validar el formato MOT
    validate_mot(seq, lines)

