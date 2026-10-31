"""RGB-only learned frontend + native ORB-SLAM3 map and optimization backend."""
from pathlib import Path
import argparse,csv,json,time,subprocess,hashlib
import cv2
import numpy as np
import torch
from dynamic_gate import DynamicPointCNN
from mast3r_frontend import MASt3RFrontend,candidates,mutual_matches,rigid_pose,transform,gate_features
ROOT=Path(__file__).resolve().parent

class NativeBackend:
    def __init__(self,out):
        self.out=out;self.log=open(out/'backend.log','w')
        self.proc=subprocess.Popen([str(ROOT/'build/orb/mast3r_backend')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
    def command(self,op,path):
        self.proc.stdin.write(f'{op} {path}\n');self.proc.stdin.flush()
        while True:
            line=self.proc.stdout.readline()
            if not line:raise RuntimeError(f'native backend exited {self.proc.poll()}; see {self.out}/backend.log')
            self.log.write(line);self.log.flush()
            if line.startswith('ERROR '):raise RuntimeError(line)
            if line.strip()==f'DONE {path}':return
    def track(self,i,data):
        path=self.out/'packets'/f'{i:06d}.yml';path.parent.mkdir(exist_ok=True)
        fs=cv2.FileStorage(str(path),cv2.FILE_STORAGE_WRITE)
        for k,v in data.items():fs.write(k,v)
        fs.release();self.command('TRACK',path)
        fs=cv2.FileStorage(str(path)+'.result.yml',cv2.FILE_STORAGE_READ)
        result={k:fs.getNode(k).mat() for k in ('ids','world','Tcw')}
        result.update({k:fs.getNode(k).real() for k in ('valid','keyframe','keyframe_id','inliers','associated','map_points','keyframes','ba_calls','ba_edges','backend_ms','retired_map_points','local_added','relocalized','relocalizations','loop_closures','loop_inliers','global_ba_calls','fused_map_points')});fs.release()
        return result
    def save(self):self.command('SAVE',self.out)
    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write('QUIT end\n');self.proc.stdin.flush();self.proc.wait(timeout=30)
        self.log.close()

def voxel_indices(points,size):
    _,idx=np.unique(np.floor(points/size).astype(np.int64),axis=0,return_index=True)
    return np.sort(idx)
def save_ply(path,xyz,rgb):
    with open(path,'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p,c in zip(xyz,rgb):f.write('%.5f %.5f %.5f %d %d %d\n'%tuple(list(p)+list(c)))

def fuse_dense(out,clouds):
    fs=cv2.FileStorage(str(out/'keyframes.yml'),cv2.FILE_STORAGE_READ);node=fs.getNode('keyframes');poses={}
    for i in range(node.size()):
        k=node.at(i);poses[int(k.getNode('id').real())]=np.linalg.inv(k.getNode('Tcw').mat())
    fs.release();sums={};raw_count=0
    for kid,local,color in clouds:
        if kid not in poses:continue
        xyz=transform(poses[kid],local);raw_count+=len(xyz)
        # At most one vote per keyframe in each voxel.
        ix=voxel_indices(xyz,.04);xyz=xyz[ix];color=color[ix]
        for p,c in zip(xyz,color):
            key=tuple(np.floor(p/.04).astype(int));rec=sums.setdefault(key,[np.zeros(3),np.zeros(3),0]);rec[0]+=p;rec[1]+=c;rec[2]+=1
    allxyz=[];allrgb=[];xyz=[];rgb=[]
    for p,c,n in sums.values():
        allxyz.append(p/n);allrgb.append(c/n)
        if n>=2:xyz.append(p/n);rgb.append(c/n)
    save_ply(out/'dense_filtered_candidate.ply',np.asarray(allxyz).reshape(-1,3),np.asarray(allrgb).reshape(-1,3).astype(np.uint8))
    save_ply(out/'dense_static_multiview.ply',np.asarray(xyz).reshape(-1,3),np.asarray(rgb).reshape(-1,3).astype(np.uint8))
    return {'dense_samples_before_voxel':raw_count,'candidate_voxels':len(allxyz),'multiview_voxels':len(xyz),'voxel_m':.04,'minimum_distinct_keyframes':2}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,default=ROOT/'data/tum_walking_xyz_rgb')
    ap.add_argument('--frames',type=int,default=60);ap.add_argument('--stride',type=int,default=3);ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--size',type=int,default=320);ap.add_argument('--mode',choices=['baseline','dynamic'],default='dynamic');ap.add_argument('--out',type=Path)
    ap.add_argument('--test-relocalize-frame',type=int,default=-1,help='Fault injection only: clear tracking associations and require historical recovery on this frame')
    ap.add_argument('--cache',action='store_true');ap.add_argument('--calibration',type=Path,default=ROOT/'config/tum_fr3.json');args=ap.parse_args()
    out=args.out or ROOT/'results'/args.mode;out.mkdir(parents=True,exist_ok=True);(out/'visualizations').mkdir(exist_ok=True)
    config=json.loads(args.calibration.read_text());K=np.array(config['K'],np.float32)
    rows=[]
    for line in (args.dataset/'rgb.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        ts,name=line.split()[:2];rows.append((float(ts),args.dataset/name))
    rows=rows[args.start::args.stride][:args.frames];assert len(rows)>=2
    cv2.setNumThreads(2);cv2.setRNGSeed(2026);torch.manual_seed(2026)
    frontend=MASt3RFrontend(args.size);gate=DynamicPointCNN().cuda().eval()
    gate.load_state_dict(torch.load(ROOT/'checkpoints/dynamic_point_cnn.pt',map_location='cpu',weights_only=True)['model'])
    backend=NativeBackend(out);prev=None;records=[];clouds=[];valid_count=0
    cacheid=hashlib.sha256((str(args.dataset.resolve())+str(args.size)).encode()).hexdigest()[:12];cache=ROOT/'cache'/cacheid;cache.mkdir(parents=True,exist_ok=True)
    wall=time.perf_counter()
    try:
        for i,(timestamp,path) in enumerate(rows):
            start=time.perf_counter();rgb,kk=frontend.read(path,K);h,w=rgb.shape[:2]
            ref=rgb if prev is None else prev['rgb'];refname=path.name if prev is None else prev['name']
            bootstrap=prev is None and i+1<len(rows)
            target=frontend.read(rows[i+1][1],K)[0] if bootstrap else rgb
            targetname=rows[i+1][1].name if bootstrap else path.name
            cp=cache/(refname+'__'+targetname+'.npz')
            cache_hit=args.cache and cp.exists()
            if cache_hit:
                with np.load(cp) as z:pair={k:z[k].astype(np.float32) for k in z.files}
            else:
                pair=frontend.pair(ref,target)
                if args.cache:np.savez_compressed(cp,**{k:v.astype(np.float16) if isinstance(v,np.ndarray) and k.startswith('d') else v for k,v in pair.items()})
            if args.cache:
                for key in ('d1','d2','dc1','dc2'):pair[key]=pair[key].astype(np.float16).astype(np.float32)
            if bootstrap:
                # Two-view initialization: use the first view's actual pair prediction.
                # This waits for a second RGB frame, rather than pretending self-pair depth is stereo.
                for key in ('p','d','c','dc'):pair[key+'2']=pair[key+'1'].copy()
            # Anchor per-pair depth scale to tracked map points, without sensor depth or GT.
            if prev is not None:
                known=prev['ids']>=0;a=prev['uv'][known];world=prev['world'][known]
                if len(a)>=12:
                    observed=transform(prev['Tcw'],world)[:,2];predicted=pair['p1'][a[:,1],a[:,0],2]
                    ratio=observed/np.maximum(predicted,1e-5);ratio=ratio[(ratio>.3)&(ratio<3)]
                    scale=float(np.median(ratio)) if len(ratio)>=12 else 1.
                    pair['p1']*=scale;pair['p2']*=scale
            uv,score=candidates(rgb,pair['c2'],pair['dc2'])
            prevuv=uv if prev is None else prev['uv'];matches=mutual_matches(pair['d1'],pair['d2'],prevuv,uv)
            a=prevuv[matches[:,0]];b=uv[matches[:,1]]
            Trel,rigid_inliers=rigid_pose(pair['p1'][a[:,1],a[:,0]],b,kk)
            if prev is None:Trel=np.eye(4,dtype=np.float32)
            features,residual,support=gate_features(rgb,pair,prevuv,uv,matches,kk,Trel)
            torch.cuda.synchronize();gstart=time.perf_counter()
            with torch.inference_mode():
                prob=gate(torch.from_numpy(features)[None].cuda()).sigmoid()[0,0].cpu().numpy() if args.mode=='dynamic' else np.zeros((h,w),np.float32)
            torch.cuda.synchronize();gate_ms=(time.perf_counter()-gstart)*1000 if args.mode=='dynamic' else 0.
            # With no temporal support we have no motion evidence, including the initial frame.
            prob[support<.08]=0
            if prev is None:prob[:]=0
            np.savez_compressed(out/'visualizations'/f'probability_{i:04d}.npz',probability=prob,support=support)
            ids=np.full(len(uv),-1,np.int32)
            if prev is not None:ids[matches[:,1]]=prev['ids'][matches[:,0]]
            confidence_threshold=max(1.01,float(np.quantile(pair['c2'],.35)))
            valid=np.isfinite(pair['p2'][uv[:,1],uv[:,0]]).all(axis=1)&(pair['c2'][uv[:,1],uv[:,0]]>confidence_threshold)&(pair['p2'][uv[:,1],uv[:,0],2]>.1)
            dynamic=prob[uv[:,1],uv[:,0]]>.65
            retire=(ids>=0)&valid&(prob[uv[:,1],uv[:,0]]>.85)&(residual[uv[:,1],uv[:,0],0]>.5)&(residual[uv[:,1],uv[:,0],1]>.25)&(support[uv[:,1],uv[:,0]]>.3)
            rejected_ids=np.unique(ids[retire])[:,None].astype(np.int32) if args.mode=='dynamic' else np.empty((0,1),np.int32)
            keep=valid & (~dynamic if args.mode=='dynamic' else True)
            ids=ids[keep];uv=uv[keep]
            refTwc=np.eye(4,dtype=np.float32) if prev is None else np.linalg.inv(prev['Tcw'])
            xyz=transform(refTwc,pair['p2'][uv[:,1],uv[:,0]])
            # Preserve tracked IDs; voxel-thin only fresh seeds so observations stay continuous.
            old=np.flatnonzero(ids>=0);new=np.flatnonzero(ids<0);new=new[voxel_indices(xyz[new],.035)]
            sel=np.sort(np.r_[old,new]);uv=uv[sel];ids=ids[sel];xyz=xyz[sel]
            guessed=(Trel if Trel is not None else np.eye(4,dtype=np.float32))@np.linalg.inv(refTwc)
            if len(uv)>=12:
                result=backend.track(i,{'timestamp':timestamp,'width':w,'height':h,'K':kk,'uv':uv.astype(np.float32),'xyz_world':xyz.astype(np.float32),'descriptors':pair['d2'][uv[:,1],uv[:,0]].astype(np.float32),'ids':ids[:,None],'Tcw':guessed.astype(np.float32),'rejected_ids':rejected_ids,'force_relocalize':int(i==args.test_relocalize_frame)})
            else:
                result={'valid':0,'keyframe':0,'inliers':0,'ids':np.full((len(uv),1),-1,np.int32),'world':np.zeros((len(uv),3),np.float32),'Tcw':guessed,'backend_ms':0.,'ba_edges':0}
                result.update({k:records[-1][k] if records else 0 for k in ('map_points','keyframes','ba_calls')})
            if result['valid']:
                valid_count+=1
                prev={'rgb':rgb,'uv':uv,'ids':result['ids'].ravel(),'world':result['world'],'Tcw':result['Tcw'],'name':path.name}
                if result['keyframe']:
                    dense=pair['p2'];world=transform(refTwc,dense.reshape(-1,3))
                    # Pair-camera coordinates stay independent of BA/world-gauge corrections.
                    local=transform(Trel,dense.reshape(-1,3)) if Trel is not None else transform(result['Tcw'],world)
                    good=np.isfinite(local).all(axis=1)&(local[:,2]>.1)&(local[:,2]<15)&(pair['c2'].ravel()>max(1.02,float(np.quantile(pair['c2'],.5))))
                    if args.mode=='dynamic':good&=prob.ravel()<.65
                    # A full dense frame is generated; storage is subsampled by 2 pixels.
                    sampling=np.zeros((h,w),bool);sampling[::2,::2]=True;good&=sampling.ravel()
                    clouds.append((int(result['keyframe_id']),local[good],rgb.reshape(-1,3)[good]))
            # Lost: keep last successful reference and attempt a wider-baseline match next frame.
            overlay=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            heat=cv2.applyColorMap((prob*255).astype(np.uint8),cv2.COLORMAP_TURBO)
            for j,p in enumerate(uv):
                if result['ids'][j,0]>=0:cv2.circle(overlay,tuple(p),1,(0,255,0),-1)
            cv2.putText(overlay,f'{args.mode} {i} valid={int(result["valid"])} n={len(uv)}',(5,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
            cv2.imwrite(str(out/'visualizations'/f'{i:04d}.jpg'),np.hstack([overlay,heat]))
            row={'frame':i,'timestamp':timestamp,'valid':int(result['valid']),'keyframe':int(result['keyframe']),'candidates':len(keep),'cnn_rejected':int((valid&dynamic).sum()) if args.mode=='dynamic' else 0,'retained':len(uv),'pair_matches':len(matches),'rigid_inliers':int(rigid_inliers.sum()),'pose_inliers':int(result['inliers']),'map_points':int(result['map_points']),'keyframes':int(result['keyframes']),'ba_calls':int(result['ba_calls']),'ba_edges':int(result['ba_edges']),'mast3r_ms':float(pair['mast3r_ms']),'gate_ms':gate_ms,'backend_ms':result['backend_ms'],'cache_hit':int(cache_hit),'frame_wall_ms':(time.perf_counter()-start)*1000}
            row['retired_map_points']=int(result.get('retired_map_points',0))
            for k in ('local_added','relocalized','relocalizations','loop_closures','loop_inliers','global_ba_calls','fused_map_points'):row[k]=int(result.get(k,0))
            records.append(row)
            print(json.dumps(row),flush=True)
            with open(out/'frames.csv','w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=row.keys());writer.writeheader();writer.writerows(records)
        backend.save();dense_stats=fuse_dense(out,clouds)
    finally:backend.close()
    summary={'mode':args.mode,'frames':len(records),'valid_frames':valid_count,'valid_fraction':valid_count/len(records),'wall_seconds':time.perf_counter()-wall,'resolution':[w,h],'input':'RGB only; no sensor depth, GT, optical flow, or ORB extraction','dense':dense_stats,'peak_cuda_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'cnn_rejected_total':sum(r['cnn_rejected'] for r in records),'final_map_points':records[-1]['map_points'],'keyframes':records[-1]['keyframes'],'ba_calls':records[-1]['ba_calls'],'dataset':str(args.dataset),'ground_truth_used_for_estimation':False,'dynamic_accuracy':'unknown: synthetic CNN bootstrap only','frontend_source':str(ROOT/'third_party/mast3r')}
    summary.update({'pure_visual':True,'imu_used':False,'test_relocalize_frame':args.test_relocalize_frame,'local_matches_added':sum(r['local_added'] for r in records),'relocalizations':records[-1]['relocalizations'],'loop_closures':records[-1]['loop_closures'],'global_ba_calls':records[-1]['global_ba_calls'],'fused_map_points':sum(r['fused_map_points'] for r in records),'retired_map_points':sum(r['retired_map_points'] for r in records)})
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('PIPELINE_SUCCESS',json.dumps(summary),flush=True)
if __name__=='__main__':main()
