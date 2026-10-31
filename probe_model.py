from pathlib import Path
import json,numpy as np
from mast3r_frontend import MASt3RFrontend,candidates,mutual_matches,rigid_pose
root=Path(__file__).resolve().parent
seq=Path('/mnt/e/DSG_ORB_SLAM3/data/rgbd_dataset_freiburg3_walking_xyz')
rows=[l.split() for l in (seq/'rgb.txt').read_text().splitlines() if l and not l.startswith('#')]
K=np.array(json.loads((root/'config/tum_fr3.json').read_text())['K'],np.float32)
front=MASt3RFrontend();a,K=front.read(seq/rows[0][1],K);b,_=front.read(seq/rows[3][1],np.array(json.loads((root/'config/tum_fr3.json').read_text())['K'],np.float32))
p=front.pair(a,b);u,_=candidates(a,p['c1'],p['dc1']);v,_=candidates(b,p['c2'],p['dc2']);m=mutual_matches(p['d1'],p['d2'],u,v);xy=u[m[:,0]];T,ins=rigid_pose(p['p1'][xy[:,1],xy[:,0]],v[m[:,1]],K)
print('MODEL_PROBE',json.dumps({'shape':list(p['p1'].shape),'desc':list(p['d1'].shape),'ms':p['mast3r_ms'],'candidates':[len(u),len(v)],'matches':len(m),'rigid_inliers':int(ins.sum()),'pose':None if T is None else T.tolist()}),flush=True)
