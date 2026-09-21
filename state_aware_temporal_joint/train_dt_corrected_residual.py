#!/usr/bin/env python
"""Train epsilon residual after frozen dt correction."""
import argparse,json,os,sys,time,torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from state_aware_temporal_joint.time_corrected_residual import TimeCorrectedResidualNet,ResidualMeta,save_residual

class DS(Dataset):
 def __init__(self,path,split):
  r=torch.load(path,map_location='cpu',mmap=True); val=r['traj_id'].long()%5==0; self.idx=torch.where(val if split=='val' else ~val)[0]; self.r=r
 def __len__(self): return len(self.idx)
 def __getitem__(self,k):
  i=int(self.idx[k]); r=self.r; return r['x'][i].float(),r['eq'][i].float(),r['ef'][i].float(),r['t_nom'][i],r['t_corr'][i],r['is_refresh'][i].float()

def lossfn(net,b,dev):
 x,eq,ef,tn,tc,rf=[z.to(dev,non_blocking=True) for z in b]; d=net(x,eq,tn,tc,rf); final=eq+d; target=ef-eq
 mse=(d-target).pow(2).mean(); cos=(1-F.cosine_similarity(final.flatten(1),ef.flatten(1),dim=1)).mean(); return mse+.1*cos,mse,cos

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch_size',type=int,default=128); p.add_argument('--lr',type=float,default=1e-3); a=p.parse_args()
 dev=torch.device('cuda'); tr=DS(a.data,'train'); va=DS(a.data,'val'); tl=DataLoader(tr,a.batch_size,shuffle=True,drop_last=True); vl=DataLoader(va,a.batch_size)
 net=TimeCorrectedResidualNet().to(dev); opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=a.lr*.05); os.makedirs(a.output_dir,exist_ok=True); best=1e9; hist=[]
 for ep in range(1,a.epochs+1):
  net.train(); sm=0.; n=0
  for b in tl:
   l,_,_=lossfn(net,b,dev); opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1); opt.step(); sm+=float(l)*b[0].shape[0]; n+=b[0].shape[0]
  sch.step(); net.eval(); vs=vm=vc=vn=0.
  with torch.no_grad():
   for b in vl:
    l,m,c=lossfn(net,b,dev); z=b[0].shape[0]; vs+=float(l)*z;vm+=float(m)*z;vc+=float(c)*z;vn+=z
  rec={'epoch':ep,'train_loss':sm/n,'val_loss':vs/vn,'val_mse':vm/vn,'val_cos':vc/vn};hist.append(rec);print(rec,flush=True)
  if rec['val_loss']<best: best=rec['val_loss'];save_residual(a.output_dir+'/ckpt_best.pt',net,ResidualMeta())
  save_residual(a.output_dir+'/ckpt_last.pt',net,ResidualMeta());json.dump({'best':best,'history':hist},open(a.output_dir+'/history.json','w'),indent=2)
if __name__=='__main__':main()
