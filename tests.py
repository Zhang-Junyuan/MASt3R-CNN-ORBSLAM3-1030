"""Geometry, real native optimizer, and filtering contract regression tests."""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from run_pipeline import NativeBackend,voxel_indices
from mast3r_frontend import rigid_pose,project
from dynamic_gate import DynamicPointCNN
ROOT=Path(__file__).resolve().parent
def main():
    rng=np.random.default_rng(17);cv2.setRNGSeed(17)
    K=np.float32([[250,0,160],[0,250,120],[0,0,1]])
    xyz=np.column_stack([rng.uniform(-1,1,250),rng.uniform(-.7,.7,250),rng.uniform(2,4,250)]).astype(np.float32)
    T=np.eye(4,dtype=np.float32);T[0,3]=-.16
    uv,_=project(T,xyz,K);estimate,mask=rigid_pose(xyz,uv,K)
    assert np.linalg.norm(estimate-T)<1e-3
    ix=voxel_indices(np.array([[0,0,1],[.001,0,1],[.1,0,1]]),.03);assert len(ix)==2
    gate=DynamicPointCNN().eval();gate.load_state_dict(torch.load(ROOT/'checkpoints/dynamic_point_cnn.pt',weights_only=True)['model'])
    x=torch.zeros(1,10,64,80);x[:,9]=1;x[:,7,20:45,20:50]=.9;x[:,8,20:45,20:50]=.9
    with torch.no_grad():p=gate(x).sigmoid()[0,0].numpy()
    assert p[25:40,25:45].mean()>.8 and p[0:10].mean()<.2
    out=ROOT/'results/native_test';out.mkdir(parents=True,exist_ok=True);backend=NativeBackend(out)
    ids=np.full((len(xyz),1),-1,np.int32);desc=rng.normal(size=(len(xyz),24)).astype(np.float32);desc/=np.linalg.norm(desc,axis=1,keepdims=True)
    errors=[];ba=0
    try:
        for i in range(12):
            pose=np.eye(4,dtype=np.float32);pose[0,3]=-.015*i;uv,_=project(pose,xyz,K)
            noisy=uv+rng.normal(0,.08,uv.shape).astype(np.float32)
            if i>0:noisy[:15]+=rng.normal(0,12,(15,2)).astype(np.float32)
            result=backend.track(i,{'K':K,'uv':noisy,'xyz_world':xyz,'descriptors':desc,'ids':ids,'Tcw':pose,'timestamp':float(i)/10,'width':320,'height':240})
            assert result['valid']==1
            errors.append(float(np.linalg.norm(result['Tcw'][:3,3]-pose[:3,3])));ba=int(result['ba_calls'])
            # Test IDs represent fixed physical points even when a noisy observation is rejected.
            if i==0:ids=result['ids'].copy()
        result=backend.track(12,{'K':K,'uv':uv,'xyz_world':xyz,'descriptors':desc,'ids':ids,'Tcw':pose,'timestamp':1.2,'width':320,'height':240,'rejected_ids':np.int32([[ids[20,0]]])})
        assert result['retired_map_points']==1 and result['ids'][20,0]==-1
        backend.save()
    finally:backend.close()
    assert max(errors)<.03,(max(errors),errors)
    assert ba>0,'Local BA was never exercised'
    metrics={'geometry_pnp':'passed','voxel_deduplication':'passed','cnn_dynamic_region':'passed','native_pose_optimization':'passed','native_local_ba_calls':ba,'native_max_translation_error_m':max(errors),'frames':13,'persistent_map_point_retirement':'passed','no_orb_features':True}
    (ROOT/'results/tests.json').write_text(json.dumps(metrics,indent=2));print(metrics)
if __name__=='__main__':main()
