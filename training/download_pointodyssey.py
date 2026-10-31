"""Fetch the official sample archive with verified SHA256 and safe extraction."""
from pathlib import Path
import urllib.request,json,concurrent.futures,time,hashlib,tarfile
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'data/pointodyssey';DATA.mkdir(parents=True,exist_ok=True)
def request(url,headers=None):
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'MASt3R-training-research',**(headers or {})}),timeout=120)
meta=json.load(request('https://huggingface.co/api/datasets/aharley/pointodyssey'))
rev=meta['sha'];files=json.load(request(f'https://huggingface.co/api/datasets/aharley/pointodyssey/tree/{rev}?recursive=true'))
entry=next(x for x in files if x['path']=='sample.tar.gz');size=entry['size'];expected=entry['lfs']['oid']
url=f'https://huggingface.co/datasets/aharley/pointodyssey/resolve/{rev}/sample.tar.gz'
(DATA/'SOURCE.json').write_text(json.dumps({'repository':'aharley/pointodyssey','revision':rev,'file':'sample.tar.gz','size':size,'sha256':expected,'source':'official author dataset repository'},indent=2))
parts=DATA/'parts';parts.mkdir(exist_ok=True);chunk=16*1024*1024
def fetch(i):
 start=i*chunk;end=min(size-1,start+chunk-1);p=parts/f'{i:05d}.part'
 if p.exists() and p.stat().st_size==end-start+1:return
 for attempt in range(6):
  try:
   with request(url+f'?download=true&part={i}&attempt={attempt}',{'Range':f'bytes={start}-{end}'}) as r:
    assert r.status==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-'),(r.status,r.headers)
    raw=r.read();assert len(raw)==end-start+1
   p.write_bytes(raw);return
  except Exception:
   if attempt==5:raise
   time.sleep(2+attempt)
archive=DATA/'sample.tar.gz'
if not archive.exists():
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  jobs=[pool.submit(fetch,i) for i in range((size+chunk-1)//chunk)]
  for done,_ in enumerate(concurrent.futures.as_completed(jobs),1):
   _.result()
   if done%10==0:print('DOWNLOADED',done,len(jobs),flush=True)
 with archive.open('wb') as out:
  for i in range((size+chunk-1)//chunk):out.write((parts/f'{i:05d}.part').read_bytes())
h=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
assert h.hexdigest()==expected,'Archive hash mismatch'
print('ARCHIVE_VERIFIED',size,flush=True)
out=DATA/'sample';out.mkdir(exist_ok=True);names=[]
with tarfile.open(archive) as tf:
 for m in tf:
  if not m.isfile():continue
  # Keep RGB, depth, trajectory annotations, masks and metadata; skip normals/videos.
  if any(x in m.name.lower() for x in ('normal','video','.mp4')):continue
  p=(out/m.name).resolve();assert p.is_relative_to(out.resolve())
  p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists() or p.stat().st_size!=m.size:
   with tf.extractfile(m) as src,p.open('wb') as dst:
    while b:=src.read(1024*1024):dst.write(b)
  names.append(m.name)
(DATA/'EXTRACTED.json').write_text(json.dumps({'files':len(names),'annotation_files':[n for n in names if n.endswith('.npz')],'top_directories':sorted(set('/'.join(n.split('/')[:3]) for n in names))},indent=2))
print('POINTODYSSEY_READY',len(names),flush=True)
