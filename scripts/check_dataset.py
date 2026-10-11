"""Verifica el dataset UAVDT original: split de test vs carpetas de imagenes vs GT, y resume el GT.
Uso: python -I scripts/check_dataset.py data"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
root = Path(sys.argv[1]); gt_dir = root / "UAV-benchmark-MOTD_v1.0" / "GT"
test = sorted(p.name.replace(" ", "").replace("_attr.txt", "") for p in (root / "M_attr" / "test").glob("*_attr.txt"))
print(f"{len(test)} secuencias de test (M_attr/test)")
tot_f = tot_b = 0
for s in test:
    imgs = sorted((root / "UAV-benchmark-M" / s).glob("img*.jpg"))
    gt = gt_dir / f"{s}_gt.txt"
    rows = [l.split(",") for l in gt.read_text().split()] if gt.exists() else []
    maxf = max((int(r[0]) for r in rows), default=0)
    ok = "OK" if imgs and rows and maxf == len(imgs) else "REVISAR"
    tot_f += len(imgs); tot_b += len(rows)
    print(f"{s} frames={len(imgs):5d} gt_boxes={len(rows):6d} ids={len({r[1] for r in rows}):4d} gt_maxframe={maxf:5d} {ok}")
print(f"total frames={tot_f} boxes={tot_b}")
