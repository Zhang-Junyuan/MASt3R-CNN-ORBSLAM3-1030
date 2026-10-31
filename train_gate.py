"""Reproducible synthetic bootstrap; never uses evaluation sequence ground truth."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from dynamic_gate import DynamicPointCNN
ROOT=Path(__file__).resolve().parent
def sample(batch,device):
    h,w=64,80
    x=torch.rand(batch,10,h,w,device=device)
    y=torch.zeros(batch,1,h,w,device=device)
    for b in range(batch):
        if torch.rand(())<.2:continue
        a=int(torch.randint(0,40,()));c=int(torch.randint(0,50,()));dh=int(torch.randint(8,24,()));dw=int(torch.randint(8,28,()))
        y[b,0,a:a+dh,c:c+dw]=1
    support=(torch.rand(batch,1,h,w,device=device)>.3).float()
    residual=torch.rand_like(y)*.18+y*(.25+.65*torch.rand(batch,1,1,1,device=device))
    # Isolated mismatches are noise rather than dynamic supervision.
    residual=torch.where(torch.rand_like(y)<.02,torch.rand_like(y),residual)
    x[:,7:8]=residual*support;x[:,8:9]=(y*.6+torch.rand_like(y)*.25)*support;x[:,9:10]=support
    x[:,3:6]=(x[:,3:6]-.5)*2
    return x,y
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1800);args=ap.parse_args()
    torch.manual_seed(2026);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    model=DynamicPointCNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002);start=time.perf_counter()
    for step in range(args.steps):
        x,y=sample(12,device);z=model(x);loss=F.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(3.,device=device))
        opt.zero_grad();loss.backward();opt.step()
        if step%300==0:print(step,float(loss),flush=True)
    model.eval();torch.manual_seed(777);tp=fp=fn=0
    with torch.no_grad():
        for _ in range(20):
            x,y=sample(12,device);p=model(x).sigmoid()>.65;gt=y.bool()
            tp+=int((p&gt).sum());fp+=int((p&~gt).sum());fn+=int((~p&gt).sum())
    out=ROOT/'checkpoints';out.mkdir(exist_ok=True)
    torch.save({'model':model.cpu().state_dict(),'training':'procedural organized point-map residuals; NOT real dynamic labels','seed':2026},out/'dynamic_point_cnn.pt')
    metrics={'steps':args.steps,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start,'synthetic_precision':tp/max(1,tp+fp),'synthetic_recall':tp/max(1,tp+fn),'synthetic_iou':tp/max(1,tp+fp+fn),'real_dynamic_accuracy':'unknown'}
    (out/'training.json').write_text(json.dumps(metrics,indent=2));print(metrics,flush=True)
if __name__=='__main__':main()
