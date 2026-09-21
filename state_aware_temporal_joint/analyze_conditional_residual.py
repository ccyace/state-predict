#!/usr/bin/env python
"""Test conditional centering and heteroscedasticity of the remaining residual."""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from state_aware_temporal_joint.time_corrected_residual import load_residual, lookup


def ridge_fit(x, y, lam=1e-3):
    x = torch.cat([x, torch.ones(len(x), 1)], 1).double()
    y = y.double()
    scale = x[:, :-1].std(0, unbiased=False).clamp_min(1e-8)
    mean = x[:, :-1].mean(0)
    x[:, :-1] = (x[:, :-1] - mean) / scale
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0
    w = torch.linalg.solve(x.T @ x + lam * eye, x.T @ y)
    return w, mean, scale


def ridge_predict(x, fit):
    w, mean, scale = fit
    x = x.double()
    x = torch.cat([(x - mean) / scale, torch.ones(len(x), 1)], 1)
    return x @ w


def r2(y, pred):
    return float(1 - (y.double() - pred.double()).square().sum() /
                 (y.double() - y.double().mean(0)).square().sum().clamp_min(1e-20))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch_size', type=int, default=128)
    a = p.parse_args()
    dev = torch.device('cuda')
    d = torch.load(a.data, map_location='cpu', mmap=True)
    ids = d['traj_id'].long()
    # Two disjoint subsets of the held-out trajectories: fit diagnostics on %10==0,
    # report on %10==5. Neither changes the correction network.
    use = torch.where((ids % 10 == 0) | (ids % 10 == 5))[0]
    net = load_residual(a.ckpt, dev).net.eval()
    feats, means, energies, tids = [], [], [], []
    for st in range(0, len(use), a.batch_size):
        ii = use[st:st+a.batch_size]
        x = d['x'][ii].float().to(dev); eq = d['eq'][ii].float().to(dev)
        ef = d['ef'][ii].float().to(dev); tn = d['t_nom'][ii].float().to(dev)
        tc = d['t_corr'][ii].float().to(dev); rf = d['is_refresh'][ii].float().to(dev)
        z = (ef - eq) - net(x, eq, tn, tc, rf)
        # Observable state summaries only; no teacher information enters features.
        f = torch.cat([
            lookup(net.logsnr_table, tn)[:, None],
            (tc-tn)[:, None] / 20.0, rf[:, None],
            x.mean((-2,-1)), x.std((-2,-1), unbiased=False),
            eq.mean((-2,-1)), eq.std((-2,-1), unbiased=False),
            x.flatten(1).square().mean(1, keepdim=True).sqrt(),
            eq.flatten(1).square().mean(1, keepdim=True).sqrt(),
        ], 1)
        feats.append(f.cpu()); means.append(z.mean((-2,-1)).cpu())
        energies.append(z.square().mean((1,2,3)).log().cpu()[:,None]); tids.append(ids[ii])
    X=torch.cat(feats); Y=torch.cat(means); E=torch.cat(energies); T=torch.cat(tids)
    cal=T%10==0; test=T%10==5
    fm=ridge_fit(X[cal],Y[cal]); fe=ridge_fit(X[cal],E[cal])
    pm=ridge_predict(X[test],fm); pe=ridge_predict(X[test],fe)
    # Compare conditional channel-mean prediction to the unconditional-zero baseline.
    mse0=float(Y[test].double().square().mean()); mse1=float((Y[test].double()-pm).square().mean())
    # Bin by predicted log-energy; a flat curve would support time-only homoscedasticity.
    q=torch.quantile(pe[:,0],torch.linspace(0,1,6,dtype=pe.dtype)); bins=[]
    for k in range(5):
        mask=(pe[:,0]>=q[k]) & ((pe[:,0]<=q[k+1]) if k==4 else (pe[:,0]<q[k+1]))
        bins.append({'n':int(mask.sum()),'predicted_log_mse':float(pe[mask,0].mean()),
                     'observed_mse':float(E[test][mask,0].exp().mean())})
    rep={
      'split':{'calibration_states':int(cal.sum()),'test_states':int(test.sum()),
               'calibration_trajectories':len(set(T[cal].tolist())),
               'test_trajectories':len(set(T[test].tolist()))},
      'conditional_mean_probe':{'zero_baseline_channel_mean_mse':mse0,
        'ridge_channel_mean_mse':mse1,'relative_reduction':1-mse1/mse0,
        'test_r2':r2(Y[test],pm)},
      'conditional_log_variance_probe':{'test_r2':r2(E[test],pe),'quintiles':bins},
      'feature_definition':['logSNR(t)','normalized corrected-time offset','refresh flag',
        'per-channel x mean/std','per-channel epsilon mean/std','x RMS','epsilon RMS'],
      'interpretation':'Positive held-out predictability means the residual is not fully conditionally centered and/or its covariance is state dependent.'}
    os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True)
    json.dump(rep,open(a.output,'w'),indent=2)
    print(json.dumps(rep,indent=2))

if __name__=='__main__': main()
