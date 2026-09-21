#!/usr/bin/env python
"""Estimate time/channel/frequency/temporal structure of the remaining residual."""
import argparse,json,math,os,sys
from collections import defaultdict
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path[:0]=[ROOT,os.path.join(ROOT,'scripts')]
import torch
from state_aware_temporal_joint.time_corrected_residual import load_residual
from sample_diffusion_ddim import get_beta_schedule
import yaml
from calibrate_trajectory_qhat import dict2namespace

def mom():return {'n':0,'s1':0.,'s2':0.,'s3':0.,'s4':0.}
def add(m,x):
 x=x.double();m['n']+=x.numel();m['s1']+=float(x.sum());m['s2']+=float((x*x).sum());m['s3']+=float((x**3).sum());m['s4']+=float((x**4).sum())
def finish(m):
 n=max(m['n'],1);u=m['s1']/n;v=max(m['s2']/n-u*u,0);s=math.sqrt(v)
 if s<1e-12:return {'mean':u,'std':s,'skew':0.,'excess_kurtosis':0.}
 q3=m['s3']/n-3*u*m['s2']/n+2*u**3;q4=m['s4']/n-4*u*m['s3']/n+6*u*u*m['s2']/n-3*u**4
 return {'mean':u,'std':s,'skew':q3/s**3,'excess_kurtosis':q4/s**4-3}

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--ckpt',required=True);p.add_argument('--output_dir',required=True);p.add_argument('--batch_size',type=int,default=128);p.add_argument('--config',default='configs/cifar10.yml');a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True)
 d=torch.device('cuda');r=torch.load(a.data,map_location='cpu',mmap=True);ids=r['traj_id'].long();idx=torch.where(ids%5==0)[0];net=load_residual(a.ckpt,d).net.eval()
 # pass 1: sufficient statistics per nominal time
 S={}; last={}; cross=defaultdict(lambda:{'dot':0.,'a2':0.,'b2':0.,'n':0})
 for st in range(0,len(idx),a.batch_size):
  ii=idx[st:st+a.batch_size];x=r['x'][ii].float().to(d);eq=r['eq'][ii].float().to(d);ef=r['ef'][ii].float().to(d);tn=r['t_nom'][ii].float().to(d);tc=r['t_corr'][ii].float().to(d);rf=r['is_refresh'][ii].float().to(d);z=(ef-eq)-net(x,eq,tn,tc,rf)
  for k in range(len(ii)):
   t=int(tn[k]);v=z[k];q=S.setdefault(t,{'n':0,'sum':torch.zeros(3,dtype=torch.float64),'cross':torch.zeros(3,3,dtype=torch.float64),'psd':torch.zeros(17,dtype=torch.float64),'refresh':int(rf[k])});flat=v.reshape(3,-1).double().cpu();q['n']+=flat.shape[1];q['sum']+=flat.sum(1);q['cross']+=flat@flat.T
   f=torch.fft.fftshift(torch.fft.fft2(v,norm='ortho'),dim=(-2,-1)).abs().square().mean(0);h,w=f.shape;yy,xx=torch.meshgrid(torch.arange(h,device=d),torch.arange(w,device=d),indexing='ij');rb=(((yy-h//2)**2+(xx-w//2)**2).sqrt()/math.sqrt((h//2)**2+(w//2)**2)*16).floor().long().clamp(0,16)
   for b in range(17):q['psd'][b]+=float(f[rb==b].mean())
   tid=int(ids[ii[k]]);cur=v.cpu().half()
   if tid in last:
    pt,pv=last[tid];c=cross[(pt,t)];aa=pv.float();bb=cur.float();c['dot']+=float((aa*bb).sum());c['a2']+=float((aa*aa).sum());c['b2']+=float((bb*bb).sum());c['n']+=1
   last[tid]=(t,cur)
 estimates={};
 for t,q in S.items():
  mu=q['sum']/q['n'];cov=q['cross']/q['n']-mu[:,None]*mu[None,:];estimates[t]={'mean':mu.float(),'cov':cov.float(),'psd':(q['psd']/(q['n']/1024)).float(),'refresh':q['refresh']}
 # pass 2: channel whitening and channel+radial-frequency whitening diagnostics
 rawm=mom();chm=mom();fm=mom();mahal=[]
 for st in range(0,len(idx),a.batch_size):
  ii=idx[st:st+a.batch_size];x=r['x'][ii].float().to(d);eq=r['eq'][ii].float().to(d);ef=r['ef'][ii].float().to(d);tn=r['t_nom'][ii].float().to(d);tc=r['t_corr'][ii].float().to(d);rf=r['is_refresh'][ii].float().to(d);z=(ef-eq)-net(x,eq,tn,tc,rf)
  for k in range(len(ii)):
   t=int(tn[k]);e=estimates[t];mu=e['mean'].to(d);cov=e['cov'].to(d)+torch.eye(3,device=d)*1e-8;L=torch.linalg.cholesky(cov);v=z[k]-mu[:,None,None];cw=torch.linalg.solve_triangular(L,v.reshape(3,-1),upper=False).reshape_as(v);add(rawm,v);add(chm,cw);mahal.append(float(cw.square().sum(0).mean()))
   F=torch.fft.fftshift(torch.fft.fft2(cw,norm='ortho'),dim=(-2,-1));h,w=F.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=d),torch.arange(w,device=d),indexing='ij');rb=(((yy-h//2)**2+(xx-w//2)**2).sqrt()/math.sqrt((h//2)**2+(w//2)**2)*16).floor().long().clamp(0,16);ps=e['psd'].to(d).clamp_min(1e-10);scale=torch.ones_like(rb,dtype=torch.float32)
   # PSD was measured before channel whitening; radial normalization here tests shape, not exact N(0,I).
   for b in range(17):scale[rb==b]=ps[b].rsqrt()
   fw=F*scale[None];add(fm,fw.real);add(fm,fw.imag)
 cfg=dict2namespace(yaml.safe_load(open(a.config)));bn=torch.tensor(get_beta_schedule(beta_schedule=cfg.diffusion.beta_schedule,beta_start=cfg.diffusion.beta_start,beta_end=cfg.diffusion.beta_end,num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps));ab=(1-bn).cumprod(0)
 transitions={}
 for (ta,tb),c in cross.items():
  rho=c['dot']/math.sqrt(max(c['a2']*c['b2'],1e-30));at=ab[ta];an=torch.tensor(1.) if tb<0 else ab[tb];coef=float(torch.sqrt(1-an)-torch.sqrt(an/at)*torch.sqrt(1-at));cov=estimates[ta]['cov'];transitions[f'{ta}->{tb}']={'pairs':c['n'],'rho':rho,'ddim_eps_coefficient':coef,'true_increment_channel_cov':(coef*coef*cov).tolist(),'iid_model_cross_step_cov_factor':0.0,'ar1_model_cross_step_cov_factor':rho}
 rep={'split':{'samples':len(idx),'trajectories':len(set(ids[idx].tolist()))},'raw_centered':finish(rawm),'channel_whitened':finish(chm),'channel_plus_radial_frequency_whitened':finish(fm),'mean_channel_mahalanobis_energy':sum(mahal)/len(mahal),'expected_channel_mahalanobis_energy':3.0,'transitions':transitions,'per_time':{str(t):{'refresh':e['refresh'],'mean':e['mean'].tolist(),'channel_covariance':e['cov'].tolist(),'radial_psd':e['psd'].tolist()} for t,e in estimates.items()},'interpretation_notes':['IID noise matches per-step covariance only and forces all cross-step covariance to zero.','AR(1) rho is estimated from full residual-field inner products.','Frequency whitening statistic is preliminary because channel and spatial covariance are not jointly diagonal.']}
 torch.save({'per_time':estimates,'transitions':transitions},a.output_dir+'/structured_residual_estimates.pt');json.dump(rep,open(a.output_dir+'/structured_residual_analysis.json','w'),indent=2);print(json.dumps({k:rep[k] for k in ('split','raw_centered','channel_whitened','channel_plus_radial_frequency_whitened','mean_channel_mahalanobis_energy')},indent=2));print('rho range',min(v['rho'] for v in transitions.values()),max(v['rho'] for v in transitions.values()))
if __name__=='__main__':main()
