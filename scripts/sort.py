"""SORT (Bewley et al., 2016) - Simple Online and Realtime Tracking.

Idea: cada vehiculo es un "track" con un filtro de Kalman que predice donde estara en el siguiente frame.
En cada frame se emparejan las cajas predichas con las detectadas usando IoU + algoritmo hungaro.
Solo usa GEOMETRIA (posicion y tamano): no mira la apariencia del objeto.

Interfaz:
    tracker = Sort(max_age, min_hits, iou_thr)
    salida  = tracker.update(dets)
      dets   : (N, 6) [x1, y1, x2, y2, conf, cls]   detecciones del frame actual
      salida : (M, 7) [x1, y1, x2, y2, id, conf, cls]  tracks confirmados en este frame
"""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_matrix(a, b):
    """IoU entre cada caja de `a` (N,4) y cada caja de `b` (M,4), formato xyxy. Devuelve (N,M)."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


# El Kalman trabaja con z = [cx, cy, s, r]: centro, area (s = w*h) y relacion de aspecto (r = w/h).
def box_to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)]).reshape(4, 1)


def state_to_box(x):
    w = np.sqrt(max(x[2, 0] * x[3, 0], 1e-6)); h = x[2, 0] / w
    return np.array([x[0, 0] - w / 2, x[1, 0] - h / 2, x[0, 0] + w / 2, x[1, 0] + h / 2])


class Track:
    """Un vehiculo seguido. Estado Kalman (7): [cx, cy, s, r, vx, vy, vs] (posicion + velocidades; r es constante)."""
    next_id = 1

    def __init__(self, box, conf, cls):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.eye(7); kf.F[0, 4] = kf.F[1, 5] = kf.F[2, 6] = 1  # modelo de velocidad constante
        kf.H = np.eye(4, 7)                                          # solo observamos [cx, cy, s, r]
        kf.R[2:, 2:] *= 10.                # ruido de medicion: area y ratio son menos fiables
        kf.P[4:, 4:] *= 1000.; kf.P *= 10. # velocidad inicial muy incierta
        kf.Q[-1, -1] *= 0.01; kf.Q[4:, 4:] *= 0.01
        kf.x[:4] = box_to_z(box)
        self.kf, self.conf, self.cls = kf, conf, cls
        self.hits = 1            # detecciones emparejadas en total
        self.since_update = 0    # frames seguidos sin detection emparejada
        self.id = Track.next_id; Track.next_id += 1

    def predict(self):
        """Avanza el Kalman un frame y devuelve la caja predicha."""
        if self.kf.x[6, 0] + self.kf.x[2, 0] <= 0:   # evita area negativa
            self.kf.x[6, 0] = 0
        self.kf.predict(); self.since_update += 1
        return state_to_box(self.kf.x)

    def update(self, box, conf, cls):
        """Corrige el Kalman con la deteccion emparejada."""
        self.kf.update(box_to_z(box)); self.hits += 1; self.since_update = 0
        self.conf, self.cls = conf, cls


class Sort:
    def __init__(self, max_age=30, min_hits=3, iou_thr=0.3):
        """max_age : frames que un track sobrevive sin detecciones (oclusiones cortas).
        min_hits: detecciones necesarias para confirmar un track (filtra falsos positivos sueltos).
        iou_thr : IoU minimo para aceptar un emparejamiento."""
        self.max_age, self.min_hits, self.iou_thr = max_age, min_hits, iou_thr
        self.tracks, self.frame = [], 0
        Track.next_id = 1   # los IDs empiezan en 1 en cada secuencia

    def update(self, dets):
        self.frame += 1
        dets = np.asarray(dets, dtype=float).reshape(-1, 6)

        # 1) PREDECIR: donde deberia estar cada track ahora
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)

        # 2) ASOCIAR: hungaro sobre -IoU (maximiza el IoU total) y descarta pares con IoU < iou_thr
        unmatched_dets = list(range(len(dets)))
        if len(preds) and len(dets):
            iou = iou_matrix(preds, dets[:, :4])
            rows, cols = linear_sum_assignment(-iou)
            unmatched_dets = [d for d in range(len(dets)) if d not in cols]
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_thr:
                    self.tracks[r].update(dets[c, :4], dets[c, 4], dets[c, 5])   # 3) ACTUALIZAR
                else:
                    unmatched_dets.append(c)

        # 4) NACIMIENTO: cada deteccion sin pareja abre un track nuevo
        for d in unmatched_dets:
            self.tracks.append(Track(dets[d, :4], dets[d, 4], dets[d, 5]))

        # 5) SALIDA: solo tracks actualizados en ESTE frame y ya confirmados
        #    (en los primeros min_hits frames se reportan todos, porque aun no hay historial)
        out = [np.concatenate([state_to_box(t.kf.x), [t.id, t.conf, t.cls]]) for t in self.tracks
               if t.since_update == 0 and (t.hits >= self.min_hits or self.frame <= self.min_hits)]

        # 6) MUERTE: elimina tracks perdidos hace mas de max_age frames
        self.tracks = [t for t in self.tracks if t.since_update <= self.max_age]
        return np.array(out).reshape(-1, 7)
