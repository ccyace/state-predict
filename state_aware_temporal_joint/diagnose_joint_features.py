#!/usr/bin/env python
"""Rank ResNet blocks by same-input quantized-vs-float joint feature error."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import torch
import yaml

from calibrate_trajectory_qhat import dict2namespace, load_float_model, load_quant_model
from qdiff.quant_block import QuantResnetBlock


class Capture:
    def __init__(self, qnn):
        self.model = qnn.model
        self.handles = []
        self.joint = {}; self.output = {}
        for name, block in self.model.named_modules():
            if isinstance(block, QuantResnetBlock):
                self.handles.append(block.norm2.register_forward_pre_hook(self._joint(name)))
                self.handles.append(block.register_forward_hook(self._output(name)))
    def _joint(self, name):
        def hook(_m, inp): self.joint[name] = inp[0].detach()
        return hook
    def _output(self, name):
        def hook(_m, _inp, out): self.output[name] = out.detach()
        return hook
    def clear(self): self.joint = {}; self.output = {}
    def close(self):
        for h in self.handles: h.remove()


def rel_mse(a, b):
    err=(a.float()-b.float()).square().flatten(1).mean(1)
    ref=b.float().square().flatten(1).mean(1).clamp_min(1e-8)
    return float((err/ref).sum()), float(err.sum()), a.shape[0]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--samples',type=int,default=512); p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--data',default='output/noise_corr/train_data/traj_5k_fullstep_cl_cifar_w8a8.pt')
    p.add_argument('--output',default='state_aware_temporal_joint/diagnostics/joint_feature_errors.json')
    p.add_argument('--config',default='configs/cifar10.yml'); p.add_argument('--cali_ckpt',default='cifar_w8a8_ckpt.pth'); p.add_argument('--cali_data_path',default='cifar_sd1236_sample2048_allst.pt')
    p.add_argument('--split',action='store_true',default=True); p.add_argument('--quant_act',action='store_true',default=True); p.add_argument('--a_sym',action='store_true',default=True)
    p.add_argument('--weight_bit',type=int,default=8); p.add_argument('--act_bit',type=int,default=8); p.add_argument('--sm_abit',type=int,default=8); p.add_argument('--cali_st',type=int,default=10); p.add_argument('--cali_n',type=int,default=256)
    p.add_argument('--ckpt',default=''); p.add_argument('--ode_scale_json',default=''); p.add_argument('--ode_absorb_mode',default='')
    a=p.parse_args(); a.cond=False; a.joint_sa_resume=False; a.brecq_ckpt=''
    dev=torch.device('cuda'); raw=torch.load(a.data,map_location='cpu',mmap=True)
    valid=torch.where(raw['traj_id'].remainder(5)==0)[0]; g=torch.Generator().manual_seed(20260722); ids=valid[torch.randperm(len(valid),generator=g)[:a.samples]]
    with open(a.config) as f: cfg=dict2namespace(yaml.safe_load(f)); cfg.split_shortcut=a.split
    fp=load_float_model(cfg,dev,a); q=load_quant_model(cfg,dev,a,fp); del fp; cap=Capture(q); totals=defaultdict(lambda:defaultdict(float))
    with torch.no_grad():
        for bi,ch in enumerate(ids.split(a.batch_size),1):
            x=raw['x'][ch].float().to(dev); t=raw['t'][ch].float().to(dev)
            q.set_quant_state(False,False); cap.clear(); q(x,t); fj={k:v.cpu() for k,v in cap.joint.items()}; fo={k:v.cpu() for k,v in cap.output.items()}
            q.set_quant_state(True,True); cap.clear(); q(x,t)
            for name in fj:
                for kind,qa,fa in [('joint',cap.joint[name].cpu(),fj[name]),('output',cap.output[name].cpu(),fo[name])]:
                    rel,abs_,n=rel_mse(qa,fa); totals[name][kind+'_rel_sum']+=rel; totals[name][kind+'_abs_sum']+=abs_; totals[name]['n']+=n if kind=='joint' else 0
            if bi%8==0: print(f'batches {bi}/{len(list(ids.split(a.batch_size)))}',flush=True)
    rows=[]
    for name,v in totals.items():
        n=v['n']; rows.append({'block':name,'joint_rel_mse':v['joint_rel_sum']/n,'joint_mse':v['joint_abs_sum']/n,'output_rel_mse':v['output_rel_sum']/n,'output_mse':v['output_abs_sum']/n})
    rows.sort(key=lambda z:z['joint_rel_mse'],reverse=True)
    os.makedirs(os.path.dirname(a.output),exist_ok=True)
    with open(a.output,'w') as f: json.dump({'samples':len(ids),'ranking':rows,'recommended_blocks':[r['block'] for r in rows[:4]]},f,indent=2)
    print('TOP',json.dumps(rows[:8],indent=2),flush=True); cap.close()

if __name__=='__main__': main()
