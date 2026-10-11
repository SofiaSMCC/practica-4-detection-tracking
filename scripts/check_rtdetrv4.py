import sys, torch, torch.nn as nn
R="/home/marcelino.maita/courses/deep_learning/practica-4"
sys.path.insert(0, R+"/third_party/RT-DETRv4")
from engine.core import YAMLConfig
cfg=YAMLConfig(R+"/weights/rtdetrv4_visdrone_l/config.yaml")
ck=torch.load(R+"/weights/rtdetrv4_visdrone_l/model.pth",map_location="cpu",weights_only=False)
state=ck["ema"]["module"] if "ema" in ck else ck["model"]
cfg.model.load_state_dict(state)
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.model=cfg.model.deploy(); s.post=cfg.postprocessor.deploy()
    def forward(s,x,sz): return s.post(s.model(x),sz)
m=M().cuda().eval()
x=torch.rand(1,3,960,960).cuda(); sz=torch.tensor([[1024,540]]).cuda()
with torch.no_grad(): l,b,s=m(x,sz)
print("OK", l.shape,b.shape,s.shape, torch.cuda.get_device_name(0))
