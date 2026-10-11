"""Detector RT-DETRv4 (pesos VisDrone) reutilizable: carga el modelo y devuelve detecciones crudas de un frame BGR."""
import sys
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "RT-DETRv4"))   # para `engine.core`
from engine.core import YAMLConfig

# etiquetas VisDrone (desde 0): 3 car, 4 van, 5 truck, 8 bus  ->  UAVDT: 1 car, 2 truck, 3 bus
VISDRONE_TO_UAVDT = {3: 1, 4: 1, 5: 2, 8: 3}


def load_detector(size="m", device=None):
    """Devuelve detect(img_bgr, min_conf) -> (xyxy (N,4), conf (N,), label VisDrone (N,)) en pixeles de la imagen original."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    wdir = ROOT / "weights" / f"rtdetrv4_visdrone_{size}"
    cfg = YAMLConfig(str(wdir / "config.yaml"), resume=str(wdir / "model.pth"))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(wdir / "model.pth", map_location="cpu", weights_only=False)
    cfg.model.load_state_dict(ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"])

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.model = cfg.model.deploy(); self.post = cfg.postprocessor.deploy()
        def forward(self, x, sz): return self.post(self.model(x), sz)

    net = Net().to(dev).eval()
    net_size = tuple(int(s) for s in cfg.yaml_cfg.get("eval_spatial_size") or [640, 640])
    tf = T.Compose([T.Resize(net_size), T.ToTensor()])

    @torch.no_grad()
    def detect(img_bgr, min_conf=0.05):
        h, w = img_bgr.shape[:2]
        x = tf(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(dev)
        lab, box, scr = (o[0].cpu().numpy() for o in net(x, torch.tensor([[w, h]], device=dev)))
        k = scr >= min_conf
        return box[k], scr[k], lab[k].astype(int)

    return detect
