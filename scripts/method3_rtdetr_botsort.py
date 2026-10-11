"""Metodo 3 (precision): RT-DETRv4 (pesos VisDrone) + BoT-SORT con compensacion de movimiento de camara (GMC).

Ideas nuevas:
  * RT-DETR es un detector tipo Transformer: usa ATENCION GLOBAL (cada parte de la imagen "mira" a todas las demas),
    asi que entiende mejor el contexto que una CNN. Tambien es NMS-free.
  * BoT-SORT = ByteTrack + GMC. El DRON se mueve, y eso arrastra a todos los vehiculos a la vez en la imagen.
    GMC estima ese movimiento global entre frames (con optical flow) y lo resta antes de predecir/emparejar,
    asi un auto parado no parece "moverse" solo porque la camara se movio.

Pipeline por frame:   imagen --RT-DETR--> detecciones --BoT-SORT (+GMC)--> tracks con ID --> linea MOT
Salida MOT: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<vis>   (class: 1=car 2=truck 3=bus)

OJO: el modelo se entreno en VisDrone y sus etiquetas empiezan en 0 (0 peaton, 1 people, 2 bicicleta, 3 car, 4 van,
5 truck, 6 triciclo, 7 triciclo con toldo, 8 bus, 9 moto). Se convierten a UAVDT: car(3)->1, van(4)->1, truck(5)->2, bus(8)->3.
(Verificado a ojo sobre M0208: los autos del GT salen con la etiqueta 3.)
Uso: python -I scripts/method3_rtdetr_botsort.py [--size m|l] [--gmc sparseOptFlow|orb|ecc|none] [--seqs M0208] [--device cpu]
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "RT-DETRv4"))   # para `engine.core`
from common import find_sequences, Dets, tracker_args, tracks_to_mot, validate_mot, save_timing
from engine.core import YAMLConfig
from ultralytics.trackers.bot_sort import BOTSORT

VISDRONE_TO_UAVDT = {3: 1, 4: 1, 5: 2, 8: 3}   # id VisDrone -> id UAVDT; el resto (peatones, motos...) se descarta

# ----------------------------------------------------------------------------- 0. argumentos
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--root", default="data")
ap.add_argument("--seqs", nargs="*", help="secuencias (por defecto las 20 de test)")
ap.add_argument("--name", help="carpeta de salida en results/")
ap.add_argument("--size", default="m", choices=["m", "l"], help="variante de RT-DETRv4")
ap.add_argument("--device", default=None, help="'cpu' o 'cuda' (por defecto: cuda si hay)")
ap.add_argument("--gmc", help="metodo de compensacion de camara: sparseOptFlow (def.), orb, sift, ecc, none")
# Valores AJUSTADOS con scripts/tune_method3.py (primeros 300 frames de M0203, M0205, M0208, M0209, M0403);
# los de fabrica de Ultralytics (0.25 / 0.1 / 0.25) dejaban pasar demasiados falsos positivos de RT-DETR (VisDrone).
ap.add_argument("--track_high", type=float, default=0.7, help="conf minima de la 1a ronda (def. ajustado 0.7)")
ap.add_argument("--track_low", type=float, default=0.15, help="conf minima de la 2a ronda (def. ajustado 0.15)")
ap.add_argument("--new_track", type=float, default=0.7, help="conf minima para crear un track (def. ajustado 0.7)")
ap.add_argument("--buffer", type=int); ap.add_argument("--match", type=float)
ap.add_argument("--max_frames", type=int, help="solo los primeros N frames (pruebas rapidas)")
a = ap.parse_args()
dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------- 1. detector RT-DETRv4
wdir = Path("weights") / f"rtdetrv4_visdrone_{a.size}"
cfg = YAMLConfig(str(wdir / "config.yaml"), resume=str(wdir / "model.pth"))
if "HGNetv2" in cfg.yaml_cfg:
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False          # no descargar pesos del backbone: ya vienen en model.pth
ckpt = torch.load(wdir / "model.pth", map_location="cpu", weights_only=False)
cfg.model.load_state_dict(ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"])

class RTDETR(nn.Module):
    """modelo + postproceso: de la imagen (1,3,H,W) a (etiquetas, cajas xyxy en pixeles originales, scores)."""
    def __init__(self):
        super().__init__(); self.model = cfg.model.deploy(); self.post = cfg.postprocessor.deploy()
    def forward(self, x, orig_size): return self.post(self.model(x), orig_size)

net = RTDETR().to(dev).eval()
net_size = tuple(int(s) for s in cfg.yaml_cfg.get("eval_spatial_size") or [640, 640])   # (alto, ancho) de entrenamiento
to_tensor = T.Compose([T.Resize(net_size), T.ToTensor()])

@torch.no_grad()
def detect(img_bgr, min_conf):
    """Devuelve Dets (solo vehiculos, clases ya en convencion UAVDT) para una imagen BGR de OpenCV."""
    h, w = img_bgr.shape[:2]
    x = to_tensor(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(dev)
    labels, boxes, scores = (o[0].cpu().numpy() for o in net(x, torch.tensor([[w, h]], device=dev)))
    keep = (scores >= min_conf) & np.isin(labels, list(VISDRONE_TO_UAVDT))
    return Dets(boxes[keep], scores[keep], [VISDRONE_TO_UAVDT[int(l)] for l in labels[keep]])

# ----------------------------------------------------------------------------- 1b. configuracion de BoT-SORT
tcfg = tracker_args("botsort.yaml", gmc_method=a.gmc, track_buffer=a.buffer, match_thresh=a.match,
                    track_high_thresh=a.track_high, track_low_thresh=a.track_low, new_track_thresh=a.new_track)
det_conf = tcfg.track_low_thresh   # BoT-SORT tambien usa 2 rondas: hay que dejar pasar las detecciones dudosas

out_dir = Path("results") / (a.name or f"method3_rtdetr{a.size}_botsort")
out_dir.mkdir(parents=True, exist_ok=True)
timing = {}

for seq, img_dir in find_sequences(a.root, a.seqs):
    # ------------------------------------------------------------------------- 2. un tracker NUEVO por secuencia
    tracker = BOTSORT(tcfg)
    images = sorted(img_dir.glob("img*.jpg"))[: a.max_frames]
    lines, t_det, t_trk = [], 0.0, 0.0

    for frame, path in enumerate(images, start=1):
        img = cv2.imread(str(path))

        # --------------------------------------------------------------------- 3. DETECCION (solo RT-DETR)
        t0 = time.perf_counter()
        dets = detect(img, det_conf)
        if dev.startswith("cuda"): torch.cuda.synchronize()   # la GPU es asincrona: esperar antes de medir el tiempo
        t1 = time.perf_counter()

        # --------------------------------------------------------------------- 4. ASOCIACION (BoT-SORT + GMC)
        # Recibe tambien la imagen: GMC la compara con la del frame anterior para estimar el movimiento del dron.
        out = tracker.update(dets, img)                   # filas: [x1,y1,x2,y2, id, conf, clase, idx_deteccion]
        t2 = time.perf_counter()
        t_det += t1 - t0
        t_trk += t2 - t1

        # --------------------------------------------------------------------- 5. escribir en formato MOT
        lines += tracks_to_mot(frame, out)

    (out_dir / f"{seq}.txt").write_text("\n".join(lines) + "\n")
    timing[seq] = dict(frames=len(images), t_det=t_det, t_trk=t_trk, fps=len(images) / (t_det + t_trk), device=str(dev))
    print(seq, timing[seq], flush=True)
    save_timing(out_dir, timing)   # tras CADA secuencia: si el job se corta, no se pierde lo hecho
    # ------------------------------------------------------------------------- 6. validar el formato MOT
    validate_mot(seq, lines)

