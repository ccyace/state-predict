"""Residual epsilon corrector conditioned on nominal and corrected diffusion time."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_logsnr_table(n=1000):
    b = torch.linspace(1e-4, 2e-2, n)
    a = (1-b).cumprod(0).clamp(1e-8, 1-1e-8)
    return a.log() - torch.log1p(-a)


def lookup(table, t):
    t = t.float().reshape(-1).clamp(0, table.numel()-1)
    lo=t.floor().long(); hi=(lo+1).clamp(max=table.numel()-1); w=t-lo.float()
    return table[lo]*(1-w)+table[hi]*w


class FiLMBlock(nn.Module):
    def __init__(self, c=32, d=64):
        super().__init__(); self.n1=nn.GroupNorm(4,c); self.c1=nn.Conv2d(c,c,3,padding=1)
        self.n2=nn.GroupNorm(4,c); self.c2=nn.Conv2d(c,c,3,padding=1); self.f=nn.Linear(d,2*c)
    def forward(self,x,e):
        s,b=self.f(e).chunk(2,1); h=self.c1(F.silu(self.n1(x))); h=self.n2(h)
        return x+self.c2(F.silu(h*(1+s[:,:,None,None])+b[:,:,None,None]))


class TimeCorrectedResidualNet(nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('logsnr_table',default_logsnr_table(),persistent=True)
        self.inp=nn.Conv2d(6,32,3,padding=1)
        self.tm=nn.Sequential(nn.Linear(4,64),nn.SiLU(),nn.Linear(64,64))
        self.blocks=nn.ModuleList([FiLMBlock() for _ in range(3)])
        self.head=nn.Sequential(nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,3,3,padding=1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,eq,t_nom,t_corr,is_refresh):
        ln=lookup(self.logsnr_table,t_nom); lc=lookup(self.logsnr_table,t_corr)
        e=self.tm(torch.stack([ln,lc,lc-ln,is_refresh.float().reshape(-1)],1))
        h=self.inp(torch.cat([x,eq],1))
        for block in self.blocks: h=block(h,e)
        return self.head(h)


@dataclass
class ResidualMeta:
    strength: float=1.0
    r_min: float=0.8
    r_max: float=1.25
    kind: str='time_corrected_residual'


class TimeCorrectedResidual:
    def __init__(self,net,meta): self.net=net; self.meta=meta
    @torch.no_grad()
    def correct(self,eq,x,t_nom,t_corr,is_refresh,strength=None):
        s=self.meta.strength if strength is None else float(strength)
        out=eq+s*self.net(x,eq,t_nom,t_corr,is_refresh)
        no=out.flatten(1).norm(2,1).view(-1,1,1,1).clamp_min(1e-8)
        nq=eq.flatten(1).norm(2,1).view(-1,1,1,1).clamp_min(1e-8); ratio=no/nq
        scale=torch.where(ratio>self.meta.r_max,self.meta.r_max/ratio,torch.ones_like(ratio))
        scale=torch.where(ratio<self.meta.r_min,self.meta.r_min/ratio,scale)
        return out*scale


def save_residual(path,net,meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)),exist_ok=True)
    torch.save({'kind':meta.kind,'meta':asdict(meta),'state_dict':net.state_dict()},path)


def load_residual(path,device):
    p=torch.load(path,map_location=device); meta=ResidualMeta(**p['meta']); net=TimeCorrectedResidualNet().to(device)
    net.load_state_dict(p['state_dict']); net.eval(); return TimeCorrectedResidual(net,meta)
