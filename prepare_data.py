"""Copy public RGB test data read-only from the previous downloaded dataset."""
from pathlib import Path
import shutil,json,hashlib
root=Path(__file__).resolve().parent
src=Path('/mnt/e/DSG_ORB_SLAM3/data/rgbd_dataset_freiburg3_walking_xyz')
dst=root/'data/tum_walking_xyz_rgb';dst.mkdir(parents=True,exist_ok=True)
manifest=[]
for line in (src/'rgb.txt').read_text().splitlines():
    if not line or line.startswith('#'):continue
    _,name=line.split()[:2];p=dst/name;p.parent.mkdir(exist_ok=True);shutil.copy2(src/name,p)
    manifest.append({'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for name in ('rgb.txt','groundtruth.txt'):
    shutil.copy2(src/name,dst/name)
(dst/'PROVENANCE.json').write_text(json.dumps({'dataset':'TUM RGB-D freiburg3_walking_xyz','copied_from_downloaded_public_dataset':str(src),'images':len(manifest),'depth_copied':False,'gt_role':'evaluation only','source_archive_sha256':'1459e9488ac0e61a2ec80dfbc35cfb77942f6d8eabded1c8d26a70be650d0e1d','files':manifest},indent=2))
print('RGB_DATA_READY',len(manifest),flush=True)
