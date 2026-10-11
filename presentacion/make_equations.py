"""Renderiza las ecuaciones de la presentacion como PNG transparentes con fuente Computer Modern (aspecto LaTeX)
y guarda sus tamanos (pulgadas, a 18 pt) en fig/eq/eq.json. Uso: python -I presentacion/make_equations.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.rcParams["mathtext.fontset"] = "cm"
OUT = Path("presentacion/fig/eq"); OUT.mkdir(parents=True, exist_ok=True)
FS, DPI = 18, 300
EQ = {
    # asociacion
    "iou": r"$\mathrm{IoU}(A,B)=\dfrac{|A\cap B|}{|A\cup B|}$",
    "cost": r"$C_{ij}=1-\mathrm{IoU}(\hat{b}_i,\,d_j)$",
    "hungarian": r"$\min_{x_{ij}\in\{0,1\}}\ \sum_{i,j} C_{ij}\,x_{ij}\quad \mathrm{s.a.}\ \sum_j x_{ij}\leq 1,\ \sum_i x_{ij}\leq 1$",
    "accept": r"$(i,j)\ \mathrm{se\ acepta\ si}\ \ \mathrm{IoU}(\hat{b}_i,d_j)\geq \tau_{\mathrm{iou}}$",
    # kalman
    "kstate": r"$\mathbf{x}=[\,u,\ v,\ s,\ r,\ \dot{u},\ \dot{v},\ \dot{s}\,]^{\top},\qquad s=w\,h,\quad r=w/h$",
    "kpred1": r"$\hat{\mathbf{x}}_{k|k-1}=F\,\mathbf{x}_{k-1|k-1}$",
    "kpred2": r"$P_{k|k-1}=F\,P_{k-1|k-1}\,F^{\top}+Q$",
    "kgain": r"$K_k=P_{k|k-1}H^{\top}\,(H\,P_{k|k-1}H^{\top}+R)^{-1}$",
    "kupd1": r"$\mathbf{x}_{k|k}=\hat{\mathbf{x}}_{k|k-1}+K_k\,(\mathbf{z}_k-H\,\hat{\mathbf{x}}_{k|k-1})$",
    "kupd2": r"$P_{k|k}=(I-K_k H)\,P_{k|k-1}$",
    # bytetrack
    "bt_high": r"$\mathcal{D}_{\mathrm{alta}}=\{\,d:\ s_d\geq\tau_{\mathrm{alta}}\,\},\qquad \mathcal{D}_{\mathrm{baja}}=\{\,d:\ \tau_{\mathrm{baja}}<s_d<\tau_{\mathrm{alta}}\,\}$",
    "bt_fuse": r"$C_{ij}=1-\mathrm{IoU}(\hat{b}_i,d_j)\cdot s_j$",
    # botsort
    "gmc_pos": r"$\tilde{\mathbf{c}}=R\,\mathbf{c}+\mathbf{t},\qquad A=[\,R\mid \mathbf{t}\,]\in\mathbb{R}^{2\times 3}$",
    "gmc_cov": r"$\tilde{\mathbf{x}}=\mathcal{R}\,\mathbf{x}+[\mathbf{t};\mathbf{0}],\qquad \tilde{P}=\mathcal{R}\,P\,\mathcal{R}^{\top}$",
    "cos": r"$C^{\mathrm{cos}}_{ij}=\dfrac{1-\mathbf{f}_i\cdot\mathbf{f}_j}{2},\qquad \|\mathbf{f}\|_2=1$",
    "fusion": r"$\hat{C}_{ij}=\min\left(C^{\mathrm{iou}}_{ij},\ C^{\mathrm{cos}}_{ij}\right)$",
    "ema": r"$\mathbf{e}_i^{\,k}=\alpha\,\mathbf{e}_i^{\,k-1}+(1-\alpha)\,\mathbf{f}_i^{\,k},\qquad \alpha=0.9$",
    # detectores
    "attn": r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\dfrac{Q K^{\top}}{\sqrt{d_k}}\right)V$",
    "match": r"$\hat{\sigma}=\arg\min_{\sigma}\ \sum_{i}\ \mathcal{L}_{\mathrm{match}}\left(y_i,\ \hat{y}_{\sigma(i)}\right)$",
    # sahi
    "zoom": r"$\mathrm{zoom}=\dfrac{\mathrm{imgsz}}{S}=\dfrac{1024}{640}=1.6$",
    "nslices": r"$N_x=\left\lceil\dfrac{W-S}{S\,(1-o)}\right\rceil+1=\left\lceil\dfrac{1024-640}{640\cdot 0.8}\right\rceil+1=2$",
    "ios": r"$\mathrm{IOS}(A,B)=\dfrac{|A\cap B|}{\min(|A|,\,|B|)}\ \geq 0.5\ \Rightarrow\ \mathrm{fusionar}$",
    # fusion de detectores
    "fuse_box": r"$\mathbf{b}=\dfrac{\sum_{m} s_m\,\mathbf{b}_m}{\sum_{m} s_m}$",
    "fuse_score": r"$s=\dfrac{1}{M}\sum_{m=1}^{M} s_m,\qquad M=2$",
    # metricas
    "mota": r"$\mathrm{MOTA}=1-\dfrac{\mathrm{FN}+\mathrm{FP}+\mathrm{IDS}}{\mathrm{GT}}$",
    "idf1": r"$\mathrm{IDF1}=\dfrac{2\,\mathrm{IDTP}}{2\,\mathrm{IDTP}+\mathrm{IDFP}+\mathrm{IDFN}}$",
    "fps": r"$\mathrm{FPS}=\dfrac{N_{\mathrm{frames}}}{t_{\mathrm{det}}+t_{\mathrm{trk}}}$",
}
sizes = {}
for name, tex in EQ.items():
    fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, tex, fontsize=FS, color="#1A1A1A")
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.03, transparent=True); plt.close(fig)
    w, h = Image.open(p).size; sizes[name] = [round(w / DPI, 4), round(h / DPI, 4)]
json.dump(sizes, open(OUT / "eq.json", "w"), indent=1)
print(len(sizes), "ecuaciones OK")
