"""Actual MASt3R LoRA + CNN training. Scene-held-out validation and early stopping.
The two-scene PointOdyssey sample supports train/validation only; TUM stays untouched.
"""
from pathlib import Path
import sys,json,time,random,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,cv2,torch
from torch.nn import functional as F
from mast3r_frontend import MASt3RFrontend,candidates,mutual_matches,rigid_pose,gate_features
from dynamic_gate import DynamicPointCNN
from training.joint_model import install_lora,prepare_training,make_view,lora_state
OUT=ROOT/'training/runs/joint_scene_holdout';OUT.mkdir(parents=True,exist_ok=True)
def resized(path):
 im=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);h,w=im.shape[:2]
 return cv2.resize(im,(320,176)),np.array([320/w,176/h],np.float32)
class Scene:
 def __init__(self,path,kind):
  self.path=path;self.kind=kind
  if kind=='dynamic':
   self.rgb=sorted((path/'rgbs').glob('*.jpg'));z=np.load(path/'anno.npz');self.world=z['trajs_3d'];self.uv=z['trajs_2d'];self.vis=z['visibs'];self.valid=z['valids'];self.K=z['intrinsics'];self.T=z['extrinsics'];z.close()
  else:
   self.rgb=sorted((path/'results').glob('frame*.jpg'));self.T=np.loadtxt(path/'traj.txt').reshape(-1,4,4)
 def sample(self,index,gap,rng):
  j=index+gap;im1,sc1=resized(self.rgb[index]);im2,sc2=resized(self.rgb[j]);label=np.full((176,320),-1,np.float32)
  if self.kind=='dynamic':
   u1=self.uv[index]*sc1;u2=self.uv[j]*sc2;w1=self.world[index];w2=self.world[j]
   good=(self.vis[index]>0)&(self.vis[j]>0)&(self.valid[index]>0)&(self.valid[j]>0)&np.isfinite(w1).all(-1)&np.isfinite(w2).all(-1)
   speed=np.linalg.norm(w2-w1,axis=-1)/gap;y=np.where(speed<.0005,0,np.where(speed>.002,1,-1))
   x1=w1@self.T[index,:3,:3].T+self.T[index,:3,3];x2=w2@self.T[index,:3,:3].T+self.T[index,:3,3]
   K=self.K[j,:3,:3].copy();K[0]*=sc2[0];K[1]*=sc2[1]
   good&=y>=0
  else:
   depth=cv2.imread(str(self.path/'results'/f'depth{index:06d}.png'),-1).astype(np.float32)/6553.5
   depth=cv2.resize(depth,(320,176),interpolation=cv2.INTER_NEAREST)
   K=np.float32([[600*sc1[0],0,599.5*sc1[0]],[0,600*sc1[1],339.5*sc1[1]],[0,0,1]])
   yy,xx=np.mgrid[4:172:4,4:316:4];u1=np.stack((xx.ravel(),yy.ravel()),-1).astype(np.float32);z=depth[yy,xx].ravel();x1=np.column_stack(((u1[:,0]-K[0,2])/K[0,0]*z,(u1[:,1]-K[1,2])/K[1,1]*z,z))
   relative=np.linalg.inv(self.T[j])@self.T[index];current=x1@relative[:3,:3].T+relative[:3,3];q=current@K.T;u2=q[:,:2]/np.maximum(q[:,2:],1e-6);x2=x1.copy();y=np.zeros(len(u1));good=(z>.1)&(z<15)&(current[:,2]>.1);label[:]=0
   depth2=cv2.imread(str(self.path/'results'/f'depth{j:06d}.png'),-1).astype(np.float32)/6553.5;depth2=cv2.resize(depth2,(320,176),interpolation=cv2.INTER_NEAREST)
   uvclip=np.rint(u2).astype(int);uvclip[:,0]=uvclip[:,0].clip(0,319);uvclip[:,1]=uvclip[:,1].clip(0,175)
   good&=np.abs(depth2[uvclip[:,1],uvclip[:,0]]-current[:,2])<.05
  good&=(u1[:,0]>=1)&(u1[:,0]<319)&(u1[:,1]>=1)&(u1[:,1]<175)&(u2[:,0]>=1)&(u2[:,0]<319)&(u2[:,1]>=1)&(u2[:,1]<175)&(x1[:,2]>.1)&(x2[:,2]>.1)
  ix=np.flatnonzero(good)
  if len(ix)<16:raise ValueError('Insufficient valid tracks')
  if self.kind=='dynamic':
   pos=np.rint(u2[ix]).astype(int);label[pos[:,1],pos[:,0]]=y[ix]
  ix=rng.choice(ix,min(192,len(ix)),replace=False)
  return im1,im2,K.astype(np.float32),u1[ix].astype(np.float32),u2[ix].astype(np.float32),x1[ix].astype(np.float32),x2[ix].astype(np.float32),label
def sample_map(t,uv):
 h,w=t.shape[1:3];grid=uv.clone();grid[:,0]=grid[:,0]/(w-1)*2-1;grid[:,1]=grid[:,1]/(h-1)*2-1
 return F.grid_sample(t.permute(0,3,1,2),grid[None,None],align_corners=True)[0,:,0].T
def compute(model,gate,item,train=False):
 im1,im2,K,u1,u2,x1,x2,labels=item
 rgb=[torch.from_numpy(x.copy()).permute(2,0,1)[None].cuda().float()/255 for x in (im1,im2)]
 if train:
  gain=.85+random.random()*.3;rgb=[(x*gain).clamp(0,1) for x in rgb]
 with torch.autocast('cuda',dtype=torch.bfloat16):a,b=model(make_view(rgb[0]),make_view(rgb[1],1))
 pair={'p1':a['pts3d'][0].detach().float().cpu().numpy(),'p2':b['pts3d_in_other_view'][0].detach().float().cpu().numpy()}
 for key,out in [('d1',a['desc']),('d2',b['desc']),('c1',a['conf']),('c2',b['conf']),('dc1',a['desc_conf']),('dc2',b['desc_conf'])]:pair[key]=out[0].detach().float().cpu().numpy()
 v1,_=candidates(im1,pair['c1'],pair['dc1']);v2,_=candidates(im2,pair['c2'],pair['dc2']);m=mutual_matches(pair['d1'],pair['d2'],v1,v2);p=v1[m[:,0]]
 T,_=rigid_pose(pair['p1'][p[:,1],p[:,0]],v2[m[:,1]],K);features,_,support=gate_features(im2,pair,v1,v2,m,K,T)
 # Geometry/confidence channels retain gradients to LoRA; PnP/matching selection is stopped.
 feat=torch.from_numpy(features)[None].cuda();xyz=b['pts3d_in_other_view'].float();scale=xyz.detach().norm(dim=-1).median().clamp_min(.1)
 feat=torch.cat((rgb[1],(xyz/scale).clamp(-3,3).permute(0,3,1,2),b['conf'].float()[:,None].log1p().div(5).clamp(max=1),feat[:,7:]),1)
 logits=gate(feat)[0,0];target=torch.from_numpy(labels).cuda();valid=target>=0
 positives=(target[valid]>0).sum();negatives=(target[valid]==0).sum();weight=(negatives/(positives+1)).clamp(1,8).detach()
 cls=F.binary_cross_entropy_with_logits(logits[valid],target[valid],pos_weight=weight)
 uv1=torch.from_numpy(u1).cuda();uv2=torch.from_numpy(u2).cuda();gt1=torch.from_numpy(x1).cuda();gt2=torch.from_numpy(x2).cuda()
 pred1=sample_map(a['pts3d'].float(),uv1);pred2=sample_map(xyz,uv2)
 geom=F.smooth_l1_loss(pred1/gt1[:,2:].clamp_min(.1),gt1/gt1[:,2:].clamp_min(.1))+F.smooth_l1_loss(pred2/gt2[:,2:].clamp_min(.1),gt2/gt2[:,2:].clamp_min(.1))
 d1=F.normalize(sample_map(a['desc'].float(),uv1),dim=-1);d2=F.normalize(sample_map(b['desc'].float(),uv2),dim=-1);scores=d1@d2.T/.1
 idx=torch.arange(len(scores),device='cuda');desc=(F.cross_entropy(scores,idx)+F.cross_entropy(scores.T,idx))*.5
 loss=cls+geom+.05*desc
 return loss,{'classification':float(cls.detach()),'geometry':float(geom.detach()),'descriptor':float(desc.detach())},logits.detach().sigmoid().cpu().numpy(),labels,support
def save(path,data):
 tmp=path.with_suffix('.tmp');torch.save(data,tmp);os.replace(tmp,path)
def main():
 torch.manual_seed(2026);np.random.seed(2026);random.seed(2026);torch.set_num_threads(4);cv2.setNumThreads(2)
 paths=sorted((ROOT/'data/pointodyssey/sample').rglob('anno.npz'));assert len(paths)==2
 # Different underlying scenes; no adjacent-frame splitting. No dynamic test scene in this sample.
 train_dynamic=next(p.parent for p in paths if p.parent.name=='r4_new_f');val_dynamic=next(p.parent for p in paths if p.parent.name!='r4_new_f')
 scenes={'train':[Scene(train_dynamic,'dynamic'),Scene(Path('/home/yuan/data/replica/room0'),'static'),Scene(Path('/home/yuan/data/replica/office0'),'static')],'val':[Scene(val_dynamic,'dynamic'),Scene(Path('/home/yuan/data/replica/room1'),'static')]}
 config={'train_scenes':[str(s.path) for s in scenes['train']],'validation_scenes':[str(s.path) for s in scenes['val']],'test_reserved':['TUM walking_xyz (SLAM only, no dynamic point labels)','Replica room2, office4'],'dynamic_test_limitation':'Official sample contains only two independent scenes; no separate labeled dynamic test set','rank':4,'alpha':4,'steps_per_epoch':128,'validation_pairs_per_scene':12,'early_stop_patience':6,'minimum_epochs':5,'selection':'validation classification + geometry + 0.05 descriptor loss; test data never used','seed':2026}
 (OUT/'config.json').write_text(json.dumps(config,indent=2));print('DATA_SPLIT_READY',config,flush=True)
 front=MASt3RFrontend();model=front.model;names=install_lora(model);prepare_training(model);gate=DynamicPointCNN().cuda()
 opt=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.requires_grad],'lr':1e-5},{'params':gate.parameters(),'lr':3e-4}],weight_decay=.01)
 scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',patience=2,factor=.5,min_lr=1e-7)
 start=time.time();best=float('inf');bad=0;epoch=0;history=[]
 if (OUT/'last.pt').exists():
  from training.joint_model import load_lora
  state=torch.load(OUT/'last.pt',map_location='cpu',weights_only=False);load_lora(model,state['lora']);gate.load_state_dict(state['gate']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);epoch=state['epoch'];best=state['best'];bad=state['bad'];history=state['history'];print('RESUMED',epoch,flush=True)
 val=[];vrng=np.random.default_rng(777)
 for s in scenes['val']:
  for i in np.linspace(10,len(s.rgb)-12,12,dtype=int):
   try:val.append((s.kind,s.sample(int(i),3,vrng)))
   except ValueError:pass
 assert len(val)>=16
 rng=np.random.default_rng(2026+epoch)
 while True:
  epoch+=1;model.eval();gate.train();losses=[];opt.zero_grad(set_to_none=True)
  for step in range(128):
   scene=scenes['train'][0] if step%2==0 else scenes['train'][1+(step//2)%2]
   for attempt in range(30):
    try:item=scene.sample(int(rng.integers(0,len(scene.rgb)-8)),int(rng.choice([1,2,4])),rng);break
    except ValueError:continue
   else:raise RuntimeError('No valid supervised pair')
   loss,parts,_,_,_=compute(model,gate,item,True)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
   (loss/4).backward();losses.append(float(loss.detach()))
   if step%4==3:torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad]+list(gate.parameters()),1.);opt.step();opt.zero_grad(set_to_none=True)
   if step%16==0:
    status={'state':'training','epoch':epoch,'step':step,'loss':losses[-1],'best_validation_loss':best,'peak_cuda_mb':torch.cuda.max_memory_allocated()/2**20,'elapsed_seconds':time.time()-start};(OUT/'status.json').write_text(json.dumps(status,indent=2));print(status,flush=True)
  gate.eval();vl=[];tp=fp=fn=tn=0;static_fp=static_n=0
  with torch.no_grad():
   for kind,item in val:
    loss,parts,prob,target,support=compute(model,gate,item);vl.append(float(loss));prob[support<.08]=0;mask=target>=0;p=prob>.65;y=target>0
    tp+=int((p&y&mask).sum());fp+=int((p&~y&mask).sum());fn+=int((~p&y&mask).sum());tn+=int((~p&~y&mask).sum())
    if kind=='static':static_fp+=int(p.sum());static_n+=p.size
  score=float(np.mean(vl));scheduler.step(score);improved=score<best-.001
  if improved:best=score;bad=0
  else:bad+=1
  record={'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':score,'precision_at_065':tp/max(tp+fp,1),'recall_at_065':tp/max(tp+fn,1),'iou_at_065':tp/max(tp+fp+fn,1),'static_false_positive_rate_at_065':static_fp/max(static_n,1),'best':improved,'bad_epochs':bad,'elapsed_seconds':time.time()-start}
  history.append(record);state={'lora':lora_state(model),'gate':{k:v.detach().cpu() for k,v in gate.state_dict().items()},'config':config,'epoch':epoch,'best':best,'bad':bad,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'history':history,'threshold':.65}
  save(OUT/'last.pt',state)
  if improved:save(OUT/'best.pt',state)
  (OUT/'history.json').write_text(json.dumps(history,indent=2));(OUT/'status.json').write_text(json.dumps({'state':'validation_complete',**record},indent=2));print('EPOCH_COMPLETE',record,flush=True)
  if epoch>=5 and bad>=6:
   (OUT/'status.json').write_text(json.dumps({'state':'early_stopped','best_validation_loss':best,'best_epoch':min(history,key=lambda x:x['validation_loss'])['epoch'],'epochs':epoch,'note':'Best within this run and scene split, not a global optimum; deployment requires held-out SLAM evaluation'},indent=2));print('EARLY_STOPPED',epoch,best,flush=True);break
if __name__=='__main__':main()
