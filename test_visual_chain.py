"""Known-geometry return path, forced association loss, loop closure and map fusion."""
from pathlib import Path
import json
import numpy as np,cv2
from run_pipeline import NativeBackend
from mast3r_frontend import project
root=Path(__file__).resolve().parent
rng=np.random.default_rng(27);cv2.setRNGSeed(27)
K=np.float32([[250,0,160],[0,250,120],[0,0,1]])
xyz=np.column_stack((rng.uniform(-1,1,250),rng.uniform(-.7,.7,250),rng.uniform(2,4,250))).astype(np.float32)
desc=rng.normal(size=(250,24)).astype(np.float32);desc/=np.linalg.norm(desc,axis=1,keepdims=True)
out=root/'results/visual_chain_test';out.mkdir(parents=True,exist_ok=True)
backend=NativeBackend(out);ids=np.full((250,1),-1,np.int32);errors=[];rows=[]
try:
 for i in range(46):
  pose=np.eye(4,dtype=np.float32);pose[0,3]=-.008*min(i,45-i)
  uv,_=project(pose,xyz,K);uv+=rng.normal(0,.08,uv.shape).astype(np.float32)
  # Frame 10 explicitly removes associations, so successful recovery cannot rely on ID propagation.
  packet={'K':K,'uv':uv,'xyz_world':xyz,'descriptors':desc,'ids':np.full_like(ids,-1) if i==10 else ids,'Tcw':pose,'timestamp':float(i)/10,'width':320,'height':240,'force_relocalize':int(i==10)}
  r=backend.track(i,packet);assert r['valid']==1,(i,r['inliers'])
  if i==10:assert r['relocalized']==1,'Forced loss was not recovered through historical retrieval'
  ids=r['ids'];errors.append(float(np.linalg.norm(r['Tcw'][:3,3]-pose[:3,3])));rows.append({k:int(r[k]) for k in ('relocalizations','loop_closures','global_ba_calls','fused_map_points','local_added')})
 unrelated=rng.normal(size=desc.shape).astype(np.float32);unrelated/=np.linalg.norm(unrelated,axis=1,keepdims=True)
 negative=backend.track(46,{'K':K,'uv':uv,'xyz_world':xyz,'descriptors':unrelated,'ids':np.full_like(ids,-1),'Tcw':pose,'timestamp':4.6,'width':320,'height':240,'force_relocalize':1})
 assert negative['valid']==0 and negative['loop_closures']==rows[-1]['loop_closures'],'Unrelated features incorrectly accepted'
 backend.save()
finally:backend.close()
assert max(errors)<.03
assert rows[-1]['loop_closures']>=1 and rows[-1]['global_ba_calls']>=1,'No geometrically verified loop correction'
result={'frames':47,'valid_frames':46,'unrelated_scene_rejected':True,'forced_association_loss_frame':10,'relocalizations':rows[-1]['relocalizations'],'loop_closures':rows[-1]['loop_closures'],'global_ba_calls':rows[-1]['global_ba_calls'],'max_translation_error_m':max(errors),'imu_used':False,'test_type':'controlled synthetic geometry and descriptors; not a natural-loop benchmark'}
(root/'results/visual_chain_test.json').write_text(json.dumps(result,indent=2));print('VISUAL_CHAIN_TEST_PASSED',result)
