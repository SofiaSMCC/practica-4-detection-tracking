"""Ayudantes compartidos por los 4 metodos (cada metodo tiene su propio bucle visible en su archivo)."""
import json
from pathlib import Path
import numpy as np
from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace


def find_sequences(root, only=None):
    """Secuencias de TEST (split oficial = carpeta M_attr/test). Devuelve [(nombre, carpeta_de_imagenes)].
    Layout UAVDT original: <root>/UAV-benchmark-M/<seq>/img000001.jpg ..."""
    root = Path(root)
    names = sorted(p.name.replace(" ", "").replace("_attr.txt", "") for p in (root / "M_attr" / "test").glob("*_attr.txt"))
    seqs = [(s, root / "UAV-benchmark-M" / s) for s in names if not only or s in only]
    missing = [s for s, d in seqs if not d.is_dir()]
    assert not missing, f"faltan carpetas de imagenes: {missing}"
    return seqs


class Dets:
    """Detecciones de UN frame, con la interfaz que esperan los trackers de ultralytics
    (conf, cls, xywh y poder filtrar con una mascara booleana).
        xyxy: (N,4) cajas [x1,y1,x2,y2]   conf: (N,) confianza   cls: (N,) clase UAVDT (1=car, 2=truck, 3=bus)"""
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @property
    def xywh(self):   # [centro_x, centro_y, ancho, alto]
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)

    def __len__(self): return len(self.conf)
    def __getitem__(self, mask): return Dets(self.xyxy[mask], self.conf[mask], self.cls[mask])


def tracker_args(yaml_name, **overrides):
    """Carga la config por defecto de ultralytics (bytetrack.yaml / botsort.yaml); lo que pases != None la sobreescribe."""
    cfg = YAML.load(ROOT / "cfg" / "trackers" / yaml_name)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return IterableSimpleNamespace(**cfg)


def tracks_to_mot(frame, tracks):
    """tracks: (M,7+) [x1,y1,x2,y2,id,conf,cls,...] -> lineas MOT Challenge:
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,<vis>   (vis=1 en predicciones)"""
    return [f"{frame},{int(t[4])},{t[0]:.2f},{t[1]:.2f},{t[2] - t[0]:.2f},{t[3] - t[1]:.2f},{t[5]:.4f},{int(t[6])},1"
            for t in tracks]


def validate_mot(seq, lines):
    """Falla con un mensaje claro si alguna linea no cumple el formato MOT."""
    seen = set()
    for l in lines:
        c = l.split(",")
        assert len(c) == 9, f"{seq}: se esperaban 9 columnas, hay {len(c)}: {l}"
        assert int(c[0]) >= 1 and int(c[1]) >= 1 and float(c[4]) > 0 and float(c[5]) > 0, f"{seq}: linea invalida {l}"
        assert int(c[7]) in (1, 2, 3), f"{seq}: clase fuera de car/truck/bus: {l}"
        assert (c[0], c[1]) not in seen, f"{seq}: id repetido en el mismo frame: {l}"
        seen.add((c[0], c[1]))


def save_timing(out_dir, timing):
    """Guarda timing.json FUSIONANDO con lo que ya habia: asi se puede correr las secuencias en varias tandas
    (y si un job se corta, lo ya terminado no se pierde). evaluate.py evalua las secuencias que figuren aqui."""
    path = Path(out_dir) / "timing.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(timing)
    path.write_text(json.dumps(old, indent=1))
