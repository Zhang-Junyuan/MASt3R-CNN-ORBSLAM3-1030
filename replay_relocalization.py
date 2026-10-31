"""Fault-injection replay of actual MASt3R observations; no new network inference."""
from pathlib import Path
import json
import cv2,numpy as np
from run_pipeline import NativeBackend
root=Path(__file__).resolve().parent;source=root/'results/visual_v2/dynamic/packets';out=root/'results/visual_v2/relocalization_replay';out.mkdir(parents=True,exist_ok=True)
backend=NativeBackend(out);rows=[]
try:
 for i in range(21):
  fs=cv2.FileStorage(str(source/f'{i:06d}.yml'),cv2.FILE_STORAGE_READ)
  packet={k:fs.getNode(k).mat() for k in ('K','uv','xyz_world','descriptors','ids','Tcw','rejected_ids')}
  packet.update({k:fs.getNode(k).real() for k in ('timestamp','width','height')});fs.release()
  packet['width']=int(packet['width']);packet['height']=int(packet['height']);packet['force_relocalize']=int(i==20)
  if i==20:packet['ids']=np.full_like(packet['ids'],-1)
  packet={k:v for k,v in packet.items() if v is not None}
  r=backend.track(i,packet);rows.append({'frame':i,'valid':int(r['valid']),'relocalized':int(r['relocalized']),'inliers':int(r['inliers'])})
 backend.save()
finally:backend.close()
assert rows[-1]['valid']==1 and rows[-1]['relocalized']==1,rows[-1]
result={'input':'replay of actual MASt3R frame observations, not a second full RGB inference run','frames':21,'fault_frame':20,'fault':'clear associations and bypass local tracking','recovered':True,'recovery_inliers':rows[-1]['inliers'],'imu_used':False,'frames_detail':rows}
(root/'results/visual_v2/relocalization_replay.json').write_text(json.dumps(result,indent=2));print('REAL_FEATURE_RELOCALIZATION_PASSED',rows[-1])
