"""Controlled return visit using real MASt3R observations (not a natural-loop benchmark)."""
from pathlib import Path
import json
import cv2,numpy as np
from run_pipeline import NativeBackend
root=Path(__file__).resolve().parent;source=root/'results/visual_v2/dynamic/packets';out=root/'results/visual_v2/loop_replay';out.mkdir(parents=True,exist_ok=True)
backend=NativeBackend(out);r=None;returns=[];stamp=0.
try:
 for i,index in enumerate(list(range(120))+list(range(10))):
  fs=cv2.FileStorage(str(source/f'{index:06d}.yml'),cv2.FILE_STORAGE_READ)
  packet={k:fs.getNode(k).mat() for k in ('K','uv','xyz_world','descriptors','ids','Tcw','rejected_ids')}
  packet.update({k:fs.getNode(k).real() for k in ('timestamp','width','height')});fs.release()
  packet['width']=int(packet['width']);packet['height']=int(packet['height'])
  if i>=120:
   packet['timestamp']=stamp+.1;packet['ids']=np.full_like(packet['ids'],-1)
   if r is not None:packet['Tcw']=r['Tcw']
  stamp=packet['timestamp'];packet={k:v for k,v in packet.items() if v is not None}
  r=backend.track(i,packet)
  if i>=120:
   event={k:int(r[k]) for k in ('valid','inliers','relocalized','loop_closures','loop_inliers','global_ba_calls','fused_map_points')};event['frame']=i;returns.append(event);print(event,flush=True)
 backend.save()
finally:backend.close()
assert r['loop_closures']>=1 and r['global_ba_calls']>=1,'Controlled return did not produce a verified loop'
result={'input':'actual MASt3R observations; first 120 frames followed by replay of first 10 views','controlled_return':True,'natural_closed_loop_benchmark':False,'full_rgb_inference_on_return':False,'frames':130,'loop_closures':int(r['loop_closures']),'global_ba_calls':int(r['global_ba_calls']),'relocalizations':int(r['relocalizations']),'return_events':returns,'imu_used':False}
(root/'results/visual_v2/loop_replay.json').write_text(json.dumps(result,indent=2));print('REAL_FEATURE_LOOP_PASSED',result['loop_closures'],result['global_ba_calls'])
