#!/usr/bin/env python
"""Visualize whether the eta=1 mean-corrected epsilon residual is conditionally centered."""
import argparse, json, math, os, sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,ROOT)
import matplotlib.pyplot as plt
import numpy as np
import torch
from state_aware_temporal_joint.time_corrected_residual import load_residual, lookup

def cluster_mean_ci(y, ids):
    """Mean and normal 95% CI using trajectory means as independent units."""
    vals=[]
    for q in np.unique(ids):
        z=y[ids==q]
        if len(z): vals.append(z.mean(axis=0))
    a=np.asarray(vals);mu=a.mean(axis=0);se=a.std(axis=0,ddof=1)/math.sqrt(len(a)) if len(a)>1 else np.zeros_like(mu)
    return mu,mu-1.96*se,mu+1.96*se,len(a)

def ridge_fit(x,y,lam=1e-3):
    mu=x.mean(0);sd=x.std(0).clip(1e-8);z=(x-mu)/sd;z=np.c_[z,np.ones(len(z))]
    eye=np.eye(z.shape[1]);eye[-1,-1]=0;w=np.linalg.solve(z.T@z+lam*eye,z.T@y)
    return mu,sd,w

def ridge_pred(x,f):
    mu,sd,w=f;return np.c_[(x-mu)/sd,np.ones(len(x))]@w

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--ckpt',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--batch_size',type=int,default=128);a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True)
 dev=torch.device('cuda');d=torch.load(a.data,map_location='cpu',mmap=True);ids_all=d['traj_id'].long();idx=torch.where(ids_all%5==0)[0];net=load_residual(a.ckpt,dev).net.eval()
 before=[];after=[];feat=[];times=[];ids=[];pixel_sum=np.zeros(3);pixel_sq=np.zeros(3);pixel_n=0
 for st in range(0,len(idx),a.batch_size):
  ii=idx[st:st+a.batch_size];x=d['x'][ii].float().to(dev);eq=d['eq'][ii].float().to(dev);ef=d['ef'][ii].float().to(dev);tn=d['t_nom'][ii].float().to(dev);tc=d['t_corr'][ii].float().to(dev);rf=d['is_refresh'][ii].float().to(dev)
  raw=ef-eq;rem=raw-net(x,eq,tn,tc,rf);before.append(raw.mean((-2,-1)).cpu());after.append(rem.mean((-2,-1)).cpu());times.append(tn.cpu());ids.append(ids_all[ii])
  f=torch.stack([lookup(net.logsnr_table,tn),x.flatten(1).square().mean(1).sqrt(),eq.flatten(1).square().mean(1).sqrt(),(tc-tn)/20.0,rf],1);feat.append(f.cpu())
  rr=rem.double().sum((0,2,3)).cpu().numpy();ss=rem.double().square().sum((0,2,3)).cpu().numpy();pixel_sum+=rr;pixel_sq+=ss;pixel_n+=rem.shape[0]*rem.shape[2]*rem.shape[3]
 B=torch.cat(before).numpy();R=torch.cat(after).numpy();F=torch.cat(feat).numpy();T=torch.cat(times).numpy().astype(int);I=torch.cat(ids).numpy();channels=['R','G','B'];colors=['#d62728','#2ca02c','#1f77b4']
 # Figure 1: per-time means before/after, cluster CI across trajectories.
 uts=np.array(sorted(np.unique(T),reverse=True));fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
 time_stats={}
 for row,(Y,title) in enumerate([(B,'Before eta=1 mean correction'),(R,'Remaining residual after mean correction')]):
  mus=[];los=[];his=[]
  for t in uts:
   mu,lo,hi,n=cluster_mean_ci(Y[T==t],I[T==t]);mus.append(mu);los.append(lo);his.append(hi);time_stats.setdefault(str(t),{})['before' if row==0 else 'after']={'mean':mu.tolist(),'ci95_low':lo.tolist(),'ci95_high':hi.tolist(),'trajectories':n}
  mus=np.array(mus);los=np.array(los);his=np.array(his)
  for c in range(3):axs[row].plot(uts,mus[:,c],color=colors[c],label=channels[c],lw=1.5);axs[row].fill_between(uts,los[:,c],his[:,c],color=colors[c],alpha=.12)
  axs[row].axhline(0,color='black',lw=1);axs[row].set_ylabel('channel spatial mean');axs[row].set_title(title);axs[row].grid(alpha=.2)
 axs[0].legend(ncol=3);axs[1].set_xlabel('nominal diffusion timestep (reverse sampling goes right to left)');fig.tight_layout();fig.savefig(a.output_dir+'/01_per_timestep_mean_ci.png',dpi=180);plt.close(fig)
 # Figure 2: distribution of per-state channel means.
 fig,axs=plt.subplots(1,3,figsize=(12,3.5))
 for c in range(3):axs[c].hist(R[:,c],bins=80,density=True,color=colors[c],alpha=.75);axs[c].axvline(0,color='black',lw=1);axs[c].axvline(R[:,c].mean(),color='orange',lw=1.5,label=f'mean={R[:,c].mean():.2e}');axs[c].set_title(channels[c]);axs[c].set_xlabel('remaining residual spatial mean');axs[c].legend(fontsize=8)
 axs[0].set_ylabel('density');fig.tight_layout();fig.savefig(a.output_dir+'/02_residual_mean_distribution.png',dpi=180);plt.close(fig)
 # Figure 3: conditional bins for observable summaries, cluster CI.
 names=['logSNR(t)','x RMS','quantized epsilon RMS','corrected-time offset / 20'];fig,axs=plt.subplots(2,2,figsize=(11,8));conditional={}
 for k,ax in enumerate(axs.flat):
  order=np.argsort(F[:,k],kind='stable');groups=np.array_split(order,10);cent=[];conditional[names[k]]=[]
  for b,g in enumerate(groups):
   mask=np.zeros(len(F),dtype=bool);mask[g]=True;cent.append(float(F[mask,k].mean()));mu,lo,hi,n=cluster_mean_ci(R[mask],I[mask]);conditional[names[k]].append({'center':cent[-1],'mean':mu.tolist(),'ci95_low':lo.tolist(),'ci95_high':hi.tolist(),'states':int(mask.sum()),'trajectories':n})
   for c in range(3):ax.errorbar(cent[-1],mu[c],yerr=[[mu[c]-lo[c]],[hi[c]-mu[c]]],fmt='o',ms=3,color=colors[c],capsize=2)
  ax.axhline(0,color='black',lw=1);ax.set_xlabel(names[k]);ax.set_ylabel('conditional residual mean');ax.grid(alpha=.2)
 fig.suptitle('Remaining residual mean in observable-state deciles (95% trajectory-cluster CI)');fig.tight_layout();fig.savefig(a.output_dir+'/03_conditional_decile_means.png',dpi=180);plt.close(fig)
 # Figure 4: held-out ridge calibration, %10==0 fit and %10==5 test.
 cal=I%10==0;test=I%10==5;fit=ridge_fit(F[cal],R[cal]);P=ridge_pred(F[test],fit);y=R[test];mse0=float(np.mean(y*y));mse=float(np.mean((y-P)**2));r2=1-float(np.sum((y-P)**2)/np.sum((y-y.mean(0))**2));pf=P.reshape(-1);yf=y.reshape(-1);ch=np.tile(np.arange(3),len(P));edges=np.quantile(pf,np.linspace(0,1,11));fig,ax=plt.subplots(figsize=(7,5));calib=[]
 for b in range(10):
  mask=(pf>=edges[b]) & ((pf<=edges[b+1]) if b==9 else (pf<edges[b+1]));mu=float(yf[mask].mean());se=float(yf[mask].std(ddof=1)/math.sqrt(mask.sum()));xp=float(pf[mask].mean());calib.append({'predicted_mean':xp,'observed_mean':mu,'ci95_low':mu-1.96*se,'ci95_high':mu+1.96*se,'n':int(mask.sum())});ax.errorbar(xp,mu,yerr=1.96*se,fmt='o',color='#9467bd',capsize=3)
 lim=max(abs(pf).max(),abs(yf.mean()));ax.plot([-lim,lim],[-lim,lim],'--',color='gray',label='ideal calibration');ax.axhline(0,color='black',lw=1);ax.set_xlabel('ridge-predicted conditional channel mean');ax.set_ylabel('observed channel mean');ax.set_title(f'Held-out calibration: R2={r2:.3f}, relative MSE reduction={1-mse/mse0:.1%}');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(a.output_dir+'/04_probe_calibration.png',dpi=180);plt.close(fig)
 # Robust global mean CI from trajectory aggregates.
 gmu,glo,ghi,ng=cluster_mean_ci(R,I);bmu,blo,bhi,_=cluster_mean_ci(B,I);pix_mu=pixel_sum/pixel_n;pix_var=np.maximum(pixel_sq/pixel_n-pix_mu**2,0);pix_std=np.sqrt(pix_var)
 aft=[]
 for t in uts:
  q=time_stats[str(t)]['after'];aft.extend([(q['ci95_low'][c]<=0<=q['ci95_high'][c]) for c in range(3)])
 rep={'split':{'states':len(R),'trajectories':ng},'global_channel_mean':{'before':bmu.tolist(),'before_ci95_low':blo.tolist(),'before_ci95_high':bhi.tolist(),'after':gmu.tolist(),'after_ci95_low':glo.tolist(),'after_ci95_high':ghi.tolist()},'pixel_level':{'after_mean':pix_mu.tolist(),'after_std':pix_std.tolist(),'abs_mean_over_std':(np.abs(pix_mu)/pix_std).tolist()},'per_time':{'channel_time_ci_contains_zero_fraction':sum(aft)/len(aft),'statistics':time_stats},'conditional_deciles':conditional,'linear_probe':{'zero_baseline_mse':mse0,'ridge_mse':mse,'relative_reduction':1-mse/mse0,'test_r2':r2,'calibration_deciles':calib},'scope_warning':'These tests support low-order/channel-mean centering only; they do not prove the full residual field has zero conditional mean.'}
 json.dump(rep,open(a.output_dir+'/conditional_mean_statistics.json','w'),indent=2);print(json.dumps({k:rep[k] for k in ['split','global_channel_mean','pixel_level','per_time','linear_probe']},indent=2)[:8000])
if __name__=='__main__':main()
