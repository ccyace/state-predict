#!/usr/bin/env python
from __future__ import annotations
import argparse,json,os,sys,time
_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,_ROOT);sys.path.insert(0,os.path.join(_ROOT,'scripts'))
import torch,yaml
from calibrate_trajectory_qhat import dict2namespace,load_float_model,load_quant_model
from qdiff.trajectory_error import build_ddim_seq
from sample_diffusion_ddim import get_beta_schedule
from ddim.functions.denoising import compute_alpha
from qdiff.ddim_helpers import ddim_update
from state_aware_temporal_joint.joint_feature_corrector import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnostics',default='state_aware_temporal_joint/diagnostics/joint_feature_errors.json');p.add_argument('--data',default='output/noise_corr/train_data/traj_5k_fullstep_cl_cifar_w8a8.pt');p.add_argument('--output_dir',default='state_aware_temporal_joint/runs/joint_features_stage1');p.add_argument('--steps',type=int,default=3000);p.add_argument('--batch_size',type=int,default=4);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--lambda_eps',type=float,default=.2);p.add_argument('--lambda_state',type=float,default=.5);p.add_argument('--config',default='configs/cifar10.yml');p.add_argument('--cali_ckpt',default='cifar_w8a8_ckpt.pth');p.add_argument('--cali_data_path',default='cifar_sd1236_sample2048_allst.pt');p.add_argument('--split',action='store_true',default=True);p.add_argument('--quant_act',action='store_true',default=True);p.add_argument('--a_sym',action='store_true',default=True);p.add_argument('--weight_bit',type=int,default=8);p.add_argument('--act_bit',type=int,default=8);p.add_argument('--sm_abit',type=int,default=8);p.add_argument('--cali_st',type=int,default=10);p.add_argument('--cali_n',type=int,default=256);p.add_argument('--ckpt',default='');p.add_argument('--ode_scale_json',default='');p.add_argument('--ode_absorb_mode',default='');a=p.parse_args();a.cond=False;a.joint_sa_resume=False;a.brecq_ckpt=''
 os.makedirs(a.output_dir,exist_ok=True);dev=torch.device('cuda');raw=torch.load(a.data,map_location='cpu',mmap=True);traj=raw['traj_id'];train=torch.where(traj.remainder(5)!=0)[0];val=torch.where(traj.remainder(5)==0)[0]
 with open(a.config) as f:cfg=dict2namespace(yaml.safe_load(f));cfg.split_shortcut=a.split
 fp=load_float_model(cfg,dev,a);q=load_quant_model(cfg,dev,a,fp);del fp
 blocks=json.load(open(a.diagnostics))['recommended_blocks'];net=JointFeatureCorrector(discover_config(q,blocks)).to(dev);hooks=JointFeatureHooks(q,net);hooks.attach()
 for p0 in q.parameters():p0.requires_grad_(False)
 opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps,eta_min=1e-5);scaler=torch.amp.GradScaler('cuda');gen=torch.Generator().manual_seed(1234)
 betas=torch.tensor(get_beta_schedule(beta_schedule=cfg.diffusion.beta_schedule,beta_start=cfg.diffusion.beta_start,beta_end=cfg.diffusion.beta_end,num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps),device=dev,dtype=torch.float32);seq=build_ddim_seq(len(betas),100,'quad');nxt={int(i):int(j) for i,j in zip(reversed(seq),reversed([-1]+list(seq[:-1])))};hist=[];best=1e9;start=time.time()
 def step(ids,training):
  x=raw['x'][ids].float().to(dev);t=raw['t'][ids].float().to(dev);nt=torch.tensor([nxt[int(v)] for v in t],device=dev,dtype=torch.float32)
  hooks.enabled=False;q.set_quant_state(False,False)
  with torch.no_grad():ef=q(x,t);targets={k:v.detach() for k,v in hooks.features.items()}
  hooks.enabled=True;q.set_quant_state(True,True)
  with torch.autocast('cuda'):eq=q(x,t);lf=torch.stack([((hooks.features[k]-targets[k]).float().square().mean()/targets[k].float().square().mean().clamp_min(1e-8)) for k in blocks]).mean();le=((eq.float()-ef.float()).square().mean()/ef.float().square().mean().clamp_min(1e-8));at=compute_alpha(betas,t.long());an=compute_alpha(betas,nt.long());xq=ddim_update(x,eq,at,an,eta=0);xf=ddim_update(x,ef,at,an,eta=0);ls=(xq.float()-xf.float()).square().mean()/xf.float().square().mean().clamp_min(1e-8);loss=lf+a.lambda_eps*le+a.lambda_state*ls
  return loss,{'loss':float(loss.detach()),'feature':float(lf.detach()),'eps':float(le.detach()),'state':float(ls.detach())}
 for s in range(1,a.steps+1):
  ids=train[torch.randint(len(train),(a.batch_size,),generator=gen)];opt.zero_grad(set_to_none=True);loss,st=step(ids,True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(net.parameters(),1);scaler.step(opt);scaler.update();sched.step()
  if s%50==0 or s==1:print(f"step {s}/{a.steps} loss={st['loss']:.5f} feat={st['feature']:.5f} eps={st['eps']:.5f} state={st['state']:.5f} time={time.time()-start:.1f}s",flush=True)
  if s%250==0:
   net.eval();vals=[]
   with torch.no_grad():
    for ids0 in val[:128].split(a.batch_size):vals.append(step(ids0,False)[1])
   net.train();vm={k:sum(v[k] for v in vals)/len(vals) for k in vals[0]};vm['step']=s;hist.append(vm);print('VAL',vm,flush=True)
   if vm['loss']<best:best=vm['loss'];save_feature_corrector(os.path.join(a.output_dir,'ckpt_best.pt'),net,{'step':s,'val':vm,'blocks':blocks})
 save_feature_corrector(os.path.join(a.output_dir,'ckpt_last.pt'),net,{'step':a.steps,'blocks':blocks});json.dump({'blocks':blocks,'best':best,'history':hist},open(os.path.join(a.output_dir,'history.json'),'w'),indent=2);hooks.remove()
if __name__=='__main__':main()
