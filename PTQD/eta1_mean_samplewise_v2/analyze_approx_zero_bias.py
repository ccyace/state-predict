#!/usr/bin/env python
"""Core test: is time/channel residual bias negligible vs variance and eta=1 budget?"""
import argparse, json, math, os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))));sys.path[:0]=[ROOT,os.path.join(ROOT,'scripts')]
import matplotlib.pyplot as plt
import numpy as np
import torch, yaml
from calibrate_trajectory_qhat import dict2namespace
from sample_diffusion_ddim import get_beta_schedule
from state_aware_temporal_joint.time_corrected_residual import load_residual

def summary(x):
 x=np.asarray([v for v in x if np.isfinite(v)],dtype=float)
 return {'n':int(len(x)),'median':float(np.median(x)),'p90':float(np.quantile(x,.9)),'p95':float(np.quantile(x,.95)),'max':float(x.max()),'fraction_lt_0.001':float(np.mean(x<.001)),'fraction_lt_0.01':float(np.mean(x<.01)),'fraction_lt_0.05':float(np.mean(x<.05))}

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--ckpt',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--strength',type=float,default=1.0);p.add_argument('--eta',type=float,default=1.0);p.add_argument('--batch_size',type=int,default=128);p.add_argument('--config',default='configs/cifar10.yml');a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True)
 dev=torch.device('cuda');d=torch.load(a.data,map_location='cpu',mmap=True);ids=d['traj_id'].long();idx=torch.where(ids%5==0)[0];net=load_residual(a.ckpt,dev).net.eval();S={}
 for st in range(0,len(idx),a.batch_size):
  ii=idx[st:st+a.batch_size];x=d['x'][ii].float().to(dev);eq=d['eq'][ii].float().to(dev);ef=d['ef'][ii].float().to(dev);tn=d['t_nom'][ii].float().to(dev);tc=d['t_corr'][ii].float().to(dev);rf=d['is_refresh'][ii].float().to(dev);r=(ef-eq)-a.strength*net(x,eq,tn,tc,rf)
  for t in tn.unique():
   mask=tn==t;v=r[mask].double();q=S.setdefault(int(t),{'n':0,'sum':torch.zeros(3,dtype=torch.float64),'sq':torch.zeros(3,dtype=torch.float64)});q['n']+=v.shape[0]*v.shape[2]*v.shape[3];q['sum']+=v.sum((0,2,3)).cpu();q['sq']+=v.square().sum((0,2,3)).cpu()
 # Preserve actual reverse trajectory order for next-step coefficients.
 first=int(ids[idx[0]]);seq=d['t_nom'][ids==first].long().tolist();nxt=seq[1:]+[-1]
 cfg=dict2namespace(yaml.safe_load(open(a.config)));bn=torch.tensor(get_beta_schedule(beta_schedule=cfg.diffusion.beta_schedule,beta_start=cfg.diffusion.beta_start,beta_end=cfg.diffusion.beta_end,num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps));ab=(1-bn).cumprod(0)
 rows=[]
 for t,j in zip(seq,nxt):
  q=S[t];mu=(q['sum']/q['n']).numpy();mse=(q['sq']/q['n']).numpy();var=np.maximum(mse-mu*mu,0);kappa=mu*mu/np.maximum(var,1e-30);at=ab[t];an=torch.tensor(1.) if j<0 else ab[j];ratio=((1-at/an.clamp(min=at))*(1-an)/(1-at).clamp_min(1e-12)).clamp_min(0);sigma2=float(a.eta*a.eta*ratio);c2=math.sqrt(max(float(1-an)-sigma2,0));B=c2-math.sqrt(float(an/at))*math.sqrt(max(float(1-at),0));gamma=B*B*mu*mu/(sigma2+1e-30) if sigma2>0 else np.full(3,np.nan);rho_c=B*B*var/(sigma2+1e-30) if sigma2>0 else np.full(3,np.nan);rho_m=B*B*mse/(sigma2+1e-30) if sigma2>0 else np.full(3,np.nan);rel=(rho_m-rho_c)/np.maximum(rho_m,1e-30)
  rows.append({'t':t,'next_t':j,'samples_per_channel':q['n'],'mean':mu.tolist(),'centered_variance':var.tolist(),'mse':mse.tolist(),'kappa_bias_over_variance':kappa.tolist(),'sigma2':sigma2,'ddim_epsilon_coefficient':B,'gamma_bias_over_sampler_budget':gamma.tolist(),'rho_centered_variance_budget':rho_c.tolist(),'rho_mse_budget':rho_m.tolist(),'mse_budget_relative_overestimate':rel.tolist()})
 kap=np.array([r['kappa_bias_over_variance'] for r in rows]);gam=np.array([r['gamma_bias_over_sampler_budget'] for r in rows]);rc=np.array([r['rho_centered_variance_budget'] for r in rows]);rm=np.array([r['rho_mse_budget'] for r in rows]);over=np.array([r['mse_budget_relative_overestimate'] for r in rows]);ts=np.array([r['t'] for r in rows]);cols=['#d62728','#2ca02c','#1f77b4'];chs=['R','G','B']
 fig,axs=plt.subplots(3,1,figsize=(11,11),sharex=True)
 for c in range(3):axs[0].plot(ts,np.maximum(kap[:,c],1e-12),color=cols[c],label=chs[c]);axs[1].plot(ts,np.maximum(gam[:,c],1e-14),color=cols[c],label=chs[c]);axs[2].plot(ts,np.maximum(rc[:,c],1e-14),color=cols[c],label=f'{chs[c]} centered',lw=1.5);axs[2].plot(ts,np.maximum(rm[:,c],1e-14),color=cols[c],ls='--',alpha=.65,label=f'{chs[c]} MSE')
 for y,ls in [(.01,'1%'),(.05,'5%')]:axs[0].axhline(y,color='gray',ls='--',lw=1,label=ls)
 for y,ls in [(1e-3,'0.1%'),(1e-2,'1%')]:axs[1].axhline(y,color='gray',ls='--',lw=1,label=ls)
 axs[0].set_ylabel('kappa = bias energy / centered variance');axs[1].set_ylabel('gamma = bias update energy / sampler variance');axs[2].set_ylabel('VSC budget occupancy');axs[2].set_xlabel('nominal diffusion timestep');
 for ax in axs:ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
 axs[0].set_title('Is the remaining time/channel bias negligible? (eta=1, held-out trajectories)');fig.tight_layout();fig.savefig(a.output_dir+'/approx_zero_core_metrics.png',dpi=190);plt.close(fig)
 rep={'definition':{'conditioning_group':'nominal timestep x channel','residual':f'epsilon_fp - epsilon_q_dt - {a.strength} * mean_net','split':'traj_id % 5 == 0','states':int(len(idx)),'trajectories':int(ids[idx].unique().numel()),'eta':a.eta},'kappa_bias_over_centered_variance':summary(kap.ravel()),'gamma_bias_over_sampler_budget':summary(gam.ravel()),'centered_variance_budget_occupancy':summary(rc.ravel()),'mse_budget_relative_overestimate_due_to_bias':summary(over.ravel()),'threshold_decision':{'approx_zero_vs_residual_variance':bool(np.nanquantile(kap,.95)<.01),'approx_zero_vs_sampler_budget':bool(np.nanquantile(gam,.95)<1e-3),'mse_is_valid_covariance_proxy':bool(np.nanquantile(over,.95)<.01)},'per_time_channel':rows,'scope':'This establishes negligibility only for scalar time-channel bias; it does not prove pixelwise or full-state conditional centering.'}
 json.dump(rep,open(a.output_dir+'/approx_zero_core_metrics.json','w'),indent=2);print(json.dumps({k:v for k,v in rep.items() if k!='per_time_channel'},indent=2))
if __name__=='__main__':main()
