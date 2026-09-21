from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from qdiff.quant_block import QuantResnetBlock
from .adapter import module_key


@dataclass
class JointFeatureConfig:
    block_channels: Dict[str, int]
    bottleneck: int = 16
    max_residual: float = 0.05
    max_t: float = 1000.0


class JointFeatureCorrector(nn.Module):
    """Low-rank spatial correction after h + temporal projection."""
    def __init__(self, config: JointFeatureConfig):
        super().__init__(); self.config=config
        self.norms=nn.ModuleDict(); self.down=nn.ModuleDict(); self.time=nn.ModuleDict(); self.up=nn.ModuleDict()
        for name,c in config.block_channels.items():
            k=module_key(name); groups=min(8,c)
            while c%groups: groups-=1
            self.norms[k]=nn.GroupNorm(groups,c,affine=False)
            self.down[k]=nn.Conv2d(c,config.bottleneck,1)
            self.time[k]=nn.Linear(1,config.bottleneck)
            self.up[k]=nn.Conv2d(config.bottleneck,c,1)
            nn.init.zeros_(self.up[k].weight); nn.init.zeros_(self.up[k].bias)
    def forward_block(self,name,y,t):
        k=module_key(name); tn=(t.float()/self.config.max_t).view(-1,1)
        h=self.down[k](self.norms[k](y))+self.time[k](tn)[:,:,None,None]
        return y+self.config.max_residual*torch.tanh(self.up[k](F.silu(h)))


class JointFeatureHooks:
    def __init__(self,qnn,corrector):
        self.qnn=qnn; self.model=getattr(qnn,'model',qnn); self.corrector=corrector; self.enabled=True; self.t=None; self.features={}; self.handles=[]
    def attach(self):
        if self.handles:return
        self.handles.append(self.model.register_forward_pre_hook(self._start))
        chosen=set(self.corrector.config.block_channels)
        for name,b in self.model.named_modules():
            if isinstance(b,QuantResnetBlock) and name in chosen:
                self.handles.append(b.norm2.register_forward_pre_hook(self._hook(name)))
    def _start(self,_m,inp): self.t=inp[1]; self.features={}
    def _hook(self,name):
        def hook(_m,inp):
            y=inp[0]
            if self.enabled:y=self.corrector.forward_block(name,y,self.t)
            self.features[name]=y
            return (y,)
        return hook
    def remove(self):
        for h in self.handles:h.remove()
        self.handles=[]


def discover_config(qnn,blocks,bottleneck=16,max_residual=0.05):
    model=getattr(qnn,'model',qnn); found={}
    for name,b in model.named_modules():
        if isinstance(b,QuantResnetBlock) and name in blocks: found[name]=int(b.out_channels)
    missing=set(blocks)-set(found)
    if missing:raise ValueError(f'Blocks not found: {sorted(missing)}')
    return JointFeatureConfig(found,bottleneck,max_residual)


def save_feature_corrector(path,net,metadata=None):
    torch.save({'kind':'joint_state_time_feature','config':asdict(net.config),'state_dict':net.state_dict(),'metadata':metadata or {}},path)


def load_feature_corrector(path,device):
    p=torch.load(path,map_location=device)
    if p.get('kind')!='joint_state_time_feature':raise ValueError(path)
    net=JointFeatureCorrector(JointFeatureConfig(**p['config'])).to(device); net.load_state_dict(p['state_dict']); net.eval(); return net,p
