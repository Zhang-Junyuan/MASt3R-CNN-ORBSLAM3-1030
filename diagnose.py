from pathlib import Path
import numpy as np,cv2
root=Path(__file__).resolve().parent
for p in sorted((root/'cache').rglob('*.npz')):
    a=np.load(p)
    print(p.name,'z',np.quantile(a['p2'][:,:,2],[0,.1,.5,.9,1]),'conf',np.quantile(a['c2'],[0,.1,.5,.9,1]))
for p in sorted((root/'results/smoke/packets').glob('*.result.yml')):
    fs=cv2.FileStorage(str(p),cv2.FILE_STORAGE_READ);w=fs.getNode('world').mat();t=fs.getNode('Tcw').mat();ids=fs.getNode('ids').mat().ravel();c=w@t[:3,:3].T+t[:3,3]
    print(p.name,'Tcw',t,'depth',np.quantile(c[ids>=0,2],[0,.1,.5,.9,1]));fs.release()
