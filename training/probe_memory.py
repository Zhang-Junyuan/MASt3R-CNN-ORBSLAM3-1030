from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,cv2,numpy as np
from mast3r_frontend import MASt3RFrontend
from dynamic_gate import DynamicPointCNN
from training.joint_model import install_lora,prepare_training,make_view,lora_state
torch.manual_seed(2026);torch.set_num_threads(4)
front=MASt3RFrontend();model=front.model
paths=Path('/home/yuan/data/replica/room0/results')
rgb=[]
for i in (0,3):
 im=cv2.cvtColor(cv2.imread(str(paths/f'frame{i:06d}.jpg')),cv2.COLOR_BGR2RGB);im=cv2.resize(im,(320,176));rgb.append(torch.from_numpy(im.copy()).permute(2,0,1)[None].cuda().float()/255)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):reference=model(make_view(rgb[0]),make_view(rgb[1],1))[0]['pts3d'].detach().clone()
names=install_lora(model);prepare_training(model);gate=DynamicPointCNN().cuda()
optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'lr':1e-5},{'params':gate.parameters(),'lr':3e-4}])
start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
with torch.autocast('cuda',dtype=torch.bfloat16):
 a,b=model(make_view(rgb[0]),make_view(rgb[1],1));xyz=b['pts3d_in_other_view'].permute(0,3,1,2)
 features=torch.cat((rgb[1],xyz/xyz.detach().abs().mean().clamp_min(.1),b['conf'][:,None].log1p()/5,torch.zeros_like(rgb[1])),1)
 logits=gate(features);loss=logits.square().mean()+.01*a['pts3d'].square().mean()
loss.backward()
lg=sum(float(p.grad.float().abs().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None);cg=sum(float(p.grad.float().abs().sum()) for p in gate.parameters() if p.grad is not None)
assert np.isfinite(lg) and lg>0 and np.isfinite(cg) and cg>0,(lg,cg)
optimizer.step();torch.cuda.synchronize()
result={'resolution':[320,176],'adapter_modules':len(names),'lora_trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'cnn_parameters':sum(p.numel() for p in gate.parameters()),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'seconds':time.perf_counter()-start,'lora_gradient_l1':lg,'cnn_gradient_l1':cg,'zero_adapter_forward_relative_error':float((a['pts3d'].detach()-reference).abs().mean()/reference.abs().mean()),'optimizer_step':'passed'}
(ROOT/'training/memory_probe.json').write_text(json.dumps(result,indent=2));print('JOINT_BACKWARD_PASSED',result,flush=True)
