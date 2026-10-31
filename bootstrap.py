"""Download independently pinned upstream sources; never copy a local SLAM checkout."""
from pathlib import Path
import concurrent.futures, hashlib, json, time, urllib.request, tarfile, io
ROOT = Path(__file__).resolve().parent
def get(url):
    for attempt in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'research-bootstrap'}), timeout=120).read()
        except Exception:
            if attempt == 4: raise
            time.sleep(2 + attempt)
def tree(repo, rev):
    return json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1'))
def fetch_repo(repo, rev, dest, prefixes=None):
    t=tree(repo, rev); rev=t['sha']; dest=ROOT/dest; dest.mkdir(parents=True,exist_ok=True)
    entries=[e for e in t['tree'] if e['type']=='blob' and (prefixes is None or e['path'].startswith(prefixes) or e['path'] in ('CMakeLists.txt','LICENSE','README.md','.gitmodules'))]
    def fetch(e):
        p=dest/e['path']; p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            b=p.read_bytes()
            if hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==e['sha']: return
        b=get(f'https://raw.githubusercontent.com/{repo}/{rev}/{e["path"]}')
        assert hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==e['sha']
        p.write_bytes(b)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for i,_ in enumerate(pool.map(fetch,entries)):
            if i%40==0: print(repo,i,len(entries),flush=True)
    return {'repository':repo,'commit':rev,'files':len(entries),'submodules':[e for e in t['tree'] if e['type']=='commit']}
if __name__=='__main__':
    records=[]
    records.append(fetch_repo('UZ-SLAMLab/ORB_SLAM3','4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4','third_party/ORB_SLAM3',('src/','include/','Thirdparty/','cmake_modules/')))
    mast=fetch_repo('naver/mast3r','f5209afc300cec36239a7ac992263f36847bbba0','third_party/mast3r'); records.append(mast)
    dustrev=next(s['sha'] for s in mast['submodules'] if s['path']=='dust3r')
    dust=fetch_repo('naver/dust3r',dustrev,'third_party/mast3r/dust3r'); records.append(dust)
    crocorev=next(s['sha'] for s in dust['submodules'] if s['path']=='croco')
    records.append(fetch_repo('naver/croco',crocorev,'third_party/mast3r/dust3r/croco'))
    (ROOT/'SOURCES.json').write_text(json.dumps(records,indent=2))
    print('ALL_SOURCE_HASHES_VERIFIED',flush=True)
