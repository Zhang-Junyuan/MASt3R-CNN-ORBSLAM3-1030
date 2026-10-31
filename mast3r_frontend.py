from pathlib import Path
import sys,time,hashlib
import numpy as np
import cv2
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'third_party/mast3r/dust3r/croco/models/curope'))
sys.path.insert(0,str(ROOT/'third_party/mast3r'))
import mast3r.utils.path_to_dust3r
from mast3r.model import AsymmetricMASt3R

class MASt3RFrontend:
    def __init__(self,size=320,device='cuda'):
        self.device=device;self.size=size
        torch.set_num_threads(4)
        self.model=AsymmetricMASt3R.from_pretrained(str(ROOT/'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')).eval().to(device)
        assert str(ROOT/'third_party/mast3r') in sys.modules['mast3r.model'].__file__, 'Wrong source checkout imported'
        torch.set_num_threads(4)
    def read(self,path,K):
        bgr=cv2.imread(str(path));assert bgr is not None,path
        h,w=bgr.shape[:2];scale=self.size/max(h,w);nw,nh=round(w*scale),round(h*scale)
        im=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_AREA);ww=nw//16*16;hh=nh//16*16;ox=(nw-ww)//2;oy=(nh-hh)//2
        rgb=cv2.cvtColor(im[oy:oy+hh,ox:ox+ww],cv2.COLOR_BGR2RGB)
        kk=np.array(K,np.float32).copy();kk[0,:]*=nw/w;kk[1,:]*=nh/h;kk[0,2]-=ox;kk[1,2]-=oy
        return rgb,kk
    @torch.inference_mode()
    def pair(self,rgb1,rgb2):
        def view(rgb,i):
            h,w=rgb.shape[:2]
            return {'img':torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float().to(self.device)/127.5-1,'true_shape':torch.tensor([[h,w]],dtype=torch.int32),'idx':[i],'instance':[str(i)]}
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.autocast(device_type=self.device,dtype=torch.float16,enabled=self.device=='cuda'):
            a,b=self.model(view(rgb1,0),view(rgb2,1))
        torch.cuda.synchronize()
        result={
            'p1':a['pts3d'][0].float().cpu().numpy(),
            'p2':b['pts3d_in_other_view'][0].float().cpu().numpy(),
            'd1':a['desc'][0].float().cpu().numpy(),
            'd2':b['desc'][0].float().cpu().numpy(),
            'c1':a['conf'][0].float().cpu().numpy(),
            'c2':b['conf'][0].float().cpu().numpy(),
            'dc1':a['desc_conf'][0].float().cpu().numpy(),
            'dc2':b['desc_conf'][0].float().cpu().numpy(),
            'mast3r_ms':(time.perf_counter()-start)*1000}
        for key in ('p1','p2','d1','d2','c1','c2'):assert np.isfinite(result[key]).all(),key
        return result

def candidates(rgb,conf,desc_conf,cell=8):
    """One confidence/texture maximum per image cell; no FAST or ORB."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    score=np.sqrt(gx*gx+gy*gy)*np.log1p(np.minimum(conf,100))*np.sqrt(np.minimum(desc_conf,10))
    h,w=gray.shape;uv=[]
    for y in range(4,h-4,cell):
        for x in range(4,w-4,cell):
            tile=score[y:min(y+cell,h-4),x:min(x+cell,w-4)]
            dy,dx=np.unravel_index(np.argmax(tile),tile.shape);uv.append((x+dx,y+dy))
    return np.array(uv,np.int32),score

def mutual_matches(d1,d2,uv1,uv2):
    a=np.ascontiguousarray(d1[uv1[:,1],uv1[:,0]],np.float32)
    b=np.ascontiguousarray(d2[uv2[:,1],uv2[:,0]],np.float32)
    # Normalized MASt3R descriptors: exact L2, mutual and ratio filters.
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    forward=matcher.knnMatch(a,b,k=2);reverse=matcher.match(b,a)
    pairs=[]
    for row in forward:
        if len(row)<2:continue
        m,n=row
        if m.distance<.85*n.distance and m.distance<.9 and reverse[m.trainIdx].trainIdx==m.queryIdx:pairs.append((m.queryIdx,m.trainIdx))
    return np.array(pairs,np.int32).reshape(-1,2)

def rigid_pose(xyz,uv,K):
    if len(xyz)<12:return None,np.zeros(len(xyz),bool)
    ok,r,t,ins=cv2.solvePnPRansac(np.ascontiguousarray(xyz,np.float32),np.ascontiguousarray(uv,np.float32),K,None,iterationsCount=150,reprojectionError=3,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or ins is None or len(ins)<12:return None,np.zeros(len(xyz),bool)
    R,_=cv2.Rodrigues(r);T=np.eye(4,dtype=np.float32);T[:3,:3]=R;T[:3,3]=t.ravel();mask=np.zeros(len(xyz),bool);mask[ins.ravel()]=True
    return T,mask

def transform(T,xyz):return xyz@T[:3,:3].T+T[:3,3]
def project(T,xyz,K):
    x=transform(T,xyz);z=x[:,2];p=x@K.T
    return p[:,:2]/np.maximum(p[:,2:3],1e-6),z

def gate_features(rgb,pair,uv1,uv2,matches,K,T):
    h,w=rgb.shape[:2];num=np.zeros((h,w,2),np.float32);support=np.zeros((h,w),np.float32)
    if T is not None and len(matches):
        a=uv1[matches[:,0]];b=uv2[matches[:,1]];p=pair['p1'][a[:,1],a[:,0]];q=pair['p2'][b[:,1],b[:,0]]
        predicted,z=project(T,p,K);res=np.linalg.norm(predicted-b,axis=1)
        spatial=np.linalg.norm(p-q,axis=1)/np.maximum(np.linalg.norm(p,axis=1),.1)
        num[b[:,1],b[:,0],0]=np.minimum(res/8,1);num[b[:,1],b[:,0],1]=np.minimum(spatial/.12,1);support[b[:,1],b[:,0]]=1
    den=cv2.GaussianBlur(support,(0,0),3)
    residual=cv2.GaussianBlur(num,(0,0),3)/np.maximum(den[:,:,None],1e-6)
    supp=np.minimum(den*150,1)
    scale=np.median(np.linalg.norm(pair['p2'],axis=-1));xyz=np.clip(pair['p2']/max(float(scale),.1),-3,3)
    x=np.concatenate([rgb.astype(np.float32)/255,xyz,np.minimum(np.log1p(pair['c2'])/5,1)[:,:,None],residual,supp[:,:,None]],axis=-1)
    return x.transpose(2,0,1).copy(),residual,supp
