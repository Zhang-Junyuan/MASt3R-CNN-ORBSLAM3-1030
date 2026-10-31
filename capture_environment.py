from pathlib import Path
import json,platform,hashlib,importlib.metadata
import torch
import mast3r_frontend
import curope,mast3r.model,dust3r.model
root=Path(__file__).resolve().parent
packages={}
for name in ('torch','torchvision','numpy','opencv-python','timm','einops','roma','scipy','huggingface-hub'):
    try:packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:packages[name]='not separately installed'
data={'python':platform.python_version(),'platform':platform.platform(),'packages':packages,'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'sources':{'mast3r':mast3r.model.__file__,'dust3r':dust3r.model.__file__,'curope':curope.__file__},'hashes':{}}
for name in ('build/orb/mast3r_backend','checkpoints/dynamic_point_cnn.pt','native/mast3r_backend.cc','run_pipeline.py','mast3r_frontend.py'):
    data['hashes'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
assert all(str(root) in v for v in data['sources'].values()),'Old source or CUDA extension imported'
(root/'ENVIRONMENT.json').write_text(json.dumps(data,indent=2));print('ENVIRONMENT_VERIFIED')
