from pathlib import Path
import csv,json
import numpy as np,cv2
root=Path(__file__).resolve().parent;checks={}
for mode in ('baseline','dynamic'):
    out=root/'results'/mode
    summary=json.loads((out/'summary.json').read_text());rows=list(csv.DictReader((out/'frames.csv').open()))
    assert len(rows)==summary['frames']
    assert np.all(np.diff([float(r['timestamp']) for r in rows])>0)
    trajectory=np.loadtxt(out/'trajectory.txt',ndmin=2)
    assert len(trajectory)==sum(int(r['valid']) for r in rows)
    assert np.isfinite(trajectory).all()
    max_risk=0.
    if mode=='dynamic':
        for row in rows:
            i=int(row['frame']);packet=out/'packets'/f'{i:06d}.yml'
            if int(row['retained'])<12:continue
            fs=cv2.FileStorage(str(packet),cv2.FILE_STORAGE_READ);uv=np.rint(fs.getNode('uv').mat()).astype(int);d=fs.getNode('descriptors').mat();fs.release()
            assert d.shape==(len(uv),24) and np.isfinite(d).all()
            with np.load(out/'visualizations'/f'probability_{i:04d}.npz') as p:risk=p['probability'][uv[:,1],uv[:,0]]
            max_risk=max(max_risk,float(risk.max()));assert (risk<=.650001).all()
    clouds={}
    for name in ('sparse_map.ply','dense_filtered_candidate.ply','dense_static_multiview.ply'):
        lines=(out/name).read_text().splitlines();count=int(next(s for s in lines if s.startswith('element vertex')).split()[-1]);end=lines.index('end_header');assert len(lines)-end-1==count;assert count>0;clouds[name]=count
    checks[mode]={'processed_frames':len(rows),'valid_poses':len(trajectory),'finite_trajectory':True,'learned_descriptors':24,'maximum_retained_dynamic_probability':max_risk if mode=='dynamic' else None,'cnn_filter_contract':'passed' if mode=='dynamic' else 'disabled','cloud_counts':clouds,'retired_map_points':sum(int(r.get('retired_map_points',0)) for r in rows)}
checks['native_tests']=json.loads((root/'results/tests.json').read_text())
(root/'results/verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2));print('FINAL_VERIFICATION_PASSED')
