#!/usr/bin/env python
"""Compare empirical cumulative DDIM residual increment with IID/adjacent models."""
import argparse, json, os, sys
from collections import defaultdict
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,ROOT)
import torch
from state_aware_temporal_joint.time_corrected_residual import load_residual

def cov_field(fields):
    # fields: [N,C,H,W], covariance over trajectories and spatial positions.
    v=fields.permute(1,0,2,3).reshape(fields.shape[1],-1).double()
    m=v.mean(1); return v@v.T/v.shape[1]-m[:,None]*m[None,:]

@torch.no_grad()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--ckpt',required=True);p.add_argument('--estimates',required=True);p.add_argument('--output',required=True);p.add_argument('--batch_size',type=int,default=128);a=p.parse_args()
 d=torch.device('cuda');r=torch.load(a.data,map_location='cpu',mmap=True);ids=r['traj_id'].long();idx=torch.where(ids%5==0)[0];net=load_residual(a.ckpt,d).net.eval();est=torch.load(a.estimates,map_location='cpu');trans=est['transitions']
 bytraj=defaultdict(list)
 for st in range(0,len(idx),a.batch_size):
  ii=idx[st:st+a.batch_size];x=r['x'][ii].float().to(d);eq=r['eq'][ii].float().to(d);ef=r['ef'][ii].float().to(d);tn=r['t_nom'][ii].float().to(d);tc=r['t_corr'][ii].float().to(d);rf=r['is_refresh'][ii].float().to(d);z=((ef-eq)-net(x,eq,tn,tc,rf)).cpu()
  for k,j in enumerate(ii):bytraj[int(ids[j])].append((int(tn[k]),z[k]))
 cumulative=[];iid=torch.zeros(3,3,dtype=torch.float64);adj=torch.zeros_like(iid);pair_cross=defaultdict(list);used_steps=0
 for tid,seq in bytraj.items():
  # Dataset is already in reverse sampling order.
  total=torch.zeros_like(seq[0][1]); local=[]
  for n,(t,z) in enumerate(seq[:-1]):
   t2=seq[n+1][0];key=f'{t}->{t2}'
   if key not in trans:continue
   c=float(trans[key]['ddim_eps_coefficient']);total+=c*z;local.append((key,c,z))
  cumulative.append(total)
  if tid==next(iter(bytraj)):
   for key,c,z in local:
    t=int(key.split('->')[0]);iid+=c*c*est['per_time'][t]['cov'].double();used_steps+=1
  for (key1,c1,z1),(key2,c2,z2) in zip(local[:-1],local[1:]):
   pair_cross[(key1,key2)].append((z1,z2,c1,c2))
 # Adjacent cross-covariance is estimated over held-out trajectories and pixels.
 adj=iid.clone()
 for pairs in pair_cross.values():
  a1=torch.stack([q[0] for q in pairs]);a2=torch.stack([q[1] for q in pairs]);c1=pairs[0][2];c2=pairs[0][3]
  v1=a1.permute(1,0,2,3).reshape(3,-1).double();v2=a2.permute(1,0,2,3).reshape(3,-1).double();m1=v1.mean(1);m2=v2.mean(1);cross=v1@v2.T/v1.shape[1]-m1[:,None]*m2[None,:]
  adj+=c1*c2*(cross+cross.T)
 empirical=cov_field(torch.stack(cumulative));tr=lambda x:float(torch.trace(x))
 rep={'trajectories':len(cumulative),'linearization':'sum_i ddim_epsilon_coefficient_i * r_i (does not include downstream UNet Jacobians)','steps_used':used_steps,'channel_covariance':{'empirical':empirical.tolist(),'iid_time_only':iid.tolist(),'iid_plus_adjacent_cross_covariance':adj.tolist()},'trace':{'empirical':tr(empirical),'iid_time_only':tr(iid),'iid_plus_adjacent':tr(adj),'empirical_over_iid':tr(empirical)/tr(iid),'empirical_over_adjacent':tr(empirical)/tr(adj)}}
 os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True);json.dump(rep,open(a.output,'w'),indent=2);print(json.dumps(rep['trace'],indent=2))
if __name__=='__main__':main()
