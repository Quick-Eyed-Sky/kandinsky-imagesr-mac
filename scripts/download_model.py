import json, hashlib, subprocess, pathlib, concurrent.futures, sys
base='https://huggingface.co'
repo='kandinskylab/Kandinsky-ImageSR-2.0'
def get(url):
    return json.loads(subprocess.check_output(['curl','-fsSL','--retry','4',url]))
rev='7d5b5ed7def0b45b98f398c0b295a6b1658863db'
items=get(f'{base}/api/models/{repo}/tree/{rev}?recursive=true')
def download(item):
    name=item['path']
    if item['type']!='file' or name.startswith(('assets/','.')): return
    target=pathlib.Path(sys.argv[1],name); target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists() and target.stat().st_size == item['size']:
        if 'lfs' not in item:
            return
        with target.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() == item['lfs']['oid']:
                return
    tmp=target.with_suffix(target.suffix+'.part')
    subprocess.run(['curl','-fsSL','--retry','5','-C','-','-o',str(tmp),f'{base}/{repo}/resolve/{rev}/{name}'],check=True)
    assert tmp.stat().st_size==item['size']
    if 'lfs' in item:
        with tmp.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==item['lfs']['oid']
    tmp.replace(target)
    print(name,flush=True)
print(rev,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p: list(p.map(download,items))
