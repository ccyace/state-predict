#!/usr/bin/env python
"""Measure how residual-induced DDIM increment variance scales with grid step size."""
import argparse,json,math,os,sys
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,ROOT)
import torch,yaml
from state_aware_temporal_joint.time_corrected_residual import load_residual
from scripts.sample_diffusion_ddim import get_beta_schedule
from scripts.calibrate_trajectory_qhat import dict2namespace

@torch.no_grad()
def analyze(path,steps,net,ab,dev,bs=128):
 r=torch.load(path,map_location='cpu',mmap=True);ts=sorted(set(map(int,r['t_nom'].tolist())),reverse=True);nxt={t:(ts[i+1] if i+1<len(ts) else -1) for i,t in enumerate(ts)};out=[]
 ids=r['traj_id'].long();idx=torch.where(ids%5==0)[0] if steps==100 else torch.arange(len(ids))
 sums={}
 for st in range(0,len(idx),bs):
  ii=idx[st:st+bs];x=r['x'][ii].float().to(dev);eq=r['eq'][ii].float().to(dev);ef=r['ef'][ii].float().to(dev);tn=r['t_nom'][ii].float().to(dev);tc=r['t_corr'][ii].float().to(dev);rf=r['is_refresh'][ii].float().to(dev);z=(ef-eq)-net(x,eq,tn,tc,rf)
  for k in range(len(ii)):
   t=int(tn[k]);j=nxt[t];at=ab[t];an=torch.tensor(1.,device=dev) if j<0 else ab[j];c=float(torch.sqrt(1-an)-torch.sqrt(an/at)*torch.sqrt(1-at));q=sums.setdefault(t,[0,0.,c,j]);q[0]+=1;q[1]+=float(z[k].square().mean())
 for t,(n,m,c,j) in sums.items():
  h=t-j if j>=0 else max(t,1);rv=m/n;out.append({'steps':steps,'t':t,'t_next':j,'h':h,'residual_mse':rv,'ddim_coef':c,'increment_variance':c*c*rv,'t_bin':min(t//100,9)})
 return out

def slope(rows,key):
 x=torch.tensor([math.log(max(r['h'],1e-12)) for r in rows]);y=torch.tensor([math.log(max(r[key],1e-30)) for r in rows]);x=x-x.mean();return float((x*(y-y.mean())).sum()/(x*x).sum().clamp_min(1e-12))
def main():
 p=argparse.ArgumentParser();p.add_argument('--data50',required=True);p.add_argument('--data100',required=True);p.add_argument('--data250',required=True);p.add_argument('--ckpt',required=True);p.add_argument('--output',required=True);a=p.parse_args();dev=torch.device('cuda');net=load_residual(a.ckpt,dev).net.eval();cfg=dict2namespace(yaml.safe_load(open('configs/cifar10.yml')));b=torch.tensor(get_beta_schedule(beta_schedule=cfg.diffusion.beta_schedule,beta_start=cfg.diffusion.beta_start,beta_end=cfg.diffusion.beta_end,num_diffusion_timesteps=cfg.diffusion.num_diffusion_timesteps),device=dev);ab=(1-b).cumprod(0)
 rows=[]
 for path,n in ((a.data50,50),(a.data100,100),(a.data250,250)):rows+=analyze(path,n,net,ab,dev)
 bins={}
 for b in range(10):
  q=[r for r in rows if r['t_bin']==b and r['t_next']>=0];bins[str(b)]={'n_points':len(q),'residual_mse_vs_h_slope':slope(q,'residual_mse'),'increment_variance_vs_h_slope':slope(q,'increment_variance')} if len(q)>2 else {}
 rep={'rows':rows,'per_t_bin_loglog_slopes':bins,'overall':{'residual_mse_vs_h_slope':slope([r for r in rows if r['t_next']>=0],'residual_mse'),'increment_variance_vs_h_slope':slope([r for r in rows if r['t_next']>=0],'increment_variance')},'note':'A true drift residual with schedule-invariant residual variance gives increment variance scaling near h^2 locally; a diffusion increment gives scaling near h.'};os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True);json.dump(rep,open(a.output,'w'),indent=2);print(json.dumps({'overall':rep['overall'],'per_t_bin_loglog_slopes':bins},indent=2))
if __name__=='__main__':main()
