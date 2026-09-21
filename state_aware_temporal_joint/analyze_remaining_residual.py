#!/usr/bin/env python
"""Initial held-out analysis of r = (eps_q - eps_fp) - predicted conditional mean."""
import argparse, json, math, os, sys
from collections import defaultdict

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
import torch
from state_aware_temporal_joint.time_corrected_residual import load_residual


def moments(s):
    n=max(s['n'],1); mean=s['s1']/n; var=max(s['s2']/n-mean*mean,0.0); sd=math.sqrt(var)
    if sd<1e-12: return {'mean':mean,'std':sd,'skew':0.0,'excess_kurtosis':0.0}
    mu3=s['s3']/n-3*mean*(s['s2']/n)+2*mean**3
    mu4=s['s4']/n-4*mean*(s['s3']/n)+6*mean*mean*(s['s2']/n)-3*mean**4
    return {'mean':mean,'std':sd,'skew':mu3/sd**3,'excess_kurtosis':mu4/sd**4-3}


def add_stats(s,x):
    y=x.double(); s['n']+=y.numel(); s['s1']+=float(y.sum()); s['s2']+=float((y*y).sum());
    s['s3']+=float((y*y*y).sum()); s['s4']+=float((y*y*y*y).sum())


def fresh(): return {'n':0,'s1':0.,'s2':0.,'s3':0.,'s4':0.}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--ckpt',required=True); p.add_argument('--output',required=True); p.add_argument('--batch_size',type=int,default=128); p.add_argument('--test_mod',type=int,default=5); p.add_argument('--test_remainder',type=int,default=0); a=p.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); raw=torch.load(a.data,map_location='cpu',mmap=True); ids=raw['traj_id'].long(); idx=torch.where(ids%a.test_mod==a.test_remainder)[0]; corr=load_residual(a.ckpt,dev); net=corr.net.eval()
    estats,rstats=fresh(),fresh(); group=defaultdict(lambda:{'n':0,'e2':0.,'r2':0.,'rmean':0.}); channel_sum=torch.zeros(3,dtype=torch.float64); channel_sq=torch.zeros(3,dtype=torch.float64); count_img=0
    last={}; lag=defaultdict(lambda:{'n':0,'cos':0.,'energy_corr_num':0.,'e1':0.,'e2':0.}); low=high=0.; freq_n=0
    total_e2=total_r2=total_target_mean2=0.
    for st in range(0,len(idx),a.batch_size):
        ii=idx[st:st+a.batch_size]; x=raw['x'][ii].float().to(dev); eq=raw['eq'][ii].float().to(dev); ef=raw['ef'][ii].float().to(dev); tn=raw['t_nom'][ii].float().to(dev); tc=raw['t_corr'][ii].float().to(dev); rf=raw['is_refresh'][ii].float().to(dev)
        # Network was trained to predict ef-eq; m_hat in the document's sign is -delta.
        delta=net(x,eq,tn,tc,rf); target=ef-eq; rem=target-delta
        add_stats(estats,target); add_stats(rstats,rem); total_e2+=float((target*target).sum()); total_r2+=float((rem*rem).sum())
        ch=rem.double().mean((2,3)).cpu(); channel_sum+=ch.sum(0); channel_sq+=(ch*ch).sum(0); count_img+=len(ii)
        # Coarse spectral split, based on residual energy in centered FFT coordinates.
        z=torch.fft.fftshift(torch.fft.fft2(rem.float(),norm='ortho'),dim=(-2,-1)).abs().square().mean(1); h,w=z.shape[-2:]; yy,xx=torch.meshgrid(torch.arange(h,device=dev),torch.arange(w,device=dev),indexing='ij'); rad=((yy-h//2)**2+(xx-w//2)**2).sqrt(); low+=float(z[:,rad<=min(h,w)/4].sum()); high+=float(z[:,rad>min(h,w)/4].sum()); freq_n+=len(ii)
        for k in range(len(ii)):
            tv=float(tn[k]); key=f"t={tv:g}|refresh={int(rf[k].item())}"; g=group[key]; rr=rem[k]; ee=target[k]; g['n']+=1;g['e2']+=float(ee.square().mean());g['r2']+=float(rr.square().mean());g['rmean']+=float(rr.mean())
            tid=int(ids[ii[k]]); cur=rr.cpu().half(); cur_energy=float(rr.square().mean())
            if tid in last:
                prev_t,prev,prev_en=last[tid]; gap=abs(prev_t-tv); q=lag[f'{gap:g}']; q['n']+=1;q['cos']+=float(torch.nn.functional.cosine_similarity(cur.float().flatten(),prev.float().flatten(),dim=0));q['energy_corr_num']+=prev_en*cur_energy;q['e1']+=prev_en;q['e2']+=cur_energy
            last[tid]=(tv,cur,cur_energy)
    em=moments(estats); rm=moments(rstats); chmean=channel_sum/count_img; chstd=(channel_sq/count_img-chmean.square()).clamp_min(0).sqrt()
    groups={k:{'n':v['n'],'error_mse':v['e2']/v['n'],'residual_mse':v['r2']/v['n'],'residual_mean':v['rmean']/v['n'],'mse_reduction':1-v['r2']/max(v['e2'],1e-20)} for k,v in group.items()}
    lags={k:{'pairs':v['n'],'mean_cosine':v['cos']/max(v['n'],1)} for k,v in lag.items()}
    report={'split':{'rule':f'traj_id % {a.test_mod} == {a.test_remainder}','samples':len(idx),'trajectories':len(set(ids[idx].tolist()))},'raw_error_element_moments':em,'remaining_residual_element_moments':rm,'mse':{'before':total_e2/estats['n'],'after':total_r2/rstats['n'],'explained_fraction':1-total_r2/total_e2},'residual_spatial_mean_per_channel':{'mean':chmean.tolist(),'std_across_samples':chstd.tolist()},'frequency_energy':{'low':low,'high':high,'high_fraction':high/(low+high)},'adjacent_residual':lags,'per_time_refresh':groups,'notes':['Raw network output is used at strength 1.0.','No inference norm clipping is applied.','This split was used for checkpoint validation, so results are preliminary rather than a strict untouched test.']}
    os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True);json.dump(report,open(a.output,'w'),indent=2);print(json.dumps({k:report[k] for k in ('split','raw_error_element_moments','remaining_residual_element_moments','mse','residual_spatial_mean_per_channel','frequency_energy','adjacent_residual')},indent=2))
if __name__=='__main__':main()
