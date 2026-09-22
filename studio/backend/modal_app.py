"""Deploy with: modal deploy backend/modal_app.py"""
import os
import time
import modal
from pathlib import Path

app=modal.App('videotrans-studio')
image=(modal.Image.from_registry('nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04',add_python='3.11')
       .apt_install('ffmpeg','fonts-noto-core')
       .pip_install_from_requirements(str(Path(__file__).with_name('requirements.txt')))
       .add_local_python_source('backend'))
volume=modal.Volume.from_name('videotrans-studio-files',create_if_missing=True)
models=modal.Volume.from_name('videotrans-studio-models',create_if_missing=True)
records=modal.Dict.from_name('videotrans-studio-jobs',create_if_missing=True)
secrets=[modal.Secret.from_name('videotrans-studio')]
class Store:
    def get(self,key):return records.get(key,None)
    def set(self,key,value):records[key]=value
    def delete(self,key):records.pop(key,None)
    def items(self):return records.items()
    def lock(self,key):return records.put('lock:'+key,{'expires':time.time()+3600},skip_if_exists=True)
    def unlock(self,key):records.pop('lock:'+key,None)

@app.function(image=image,gpu='T4',volumes={'/data':volume,'/models':models},secrets=secrets,timeout=3600,max_containers=1)
def process_job(job_id:str,action:str):
    from backend.service import execute
    execute(Store(),'/data',job_id,action,volume.commit,volume.reload)
    models.commit()

@app.function(image=image,volumes={'/data':volume},secrets=secrets,timeout=360,max_containers=1)
@modal.asgi_app()
def web():
    from backend.service import make_app
    return make_app(Store(),'/data',lambda job,action:process_job.spawn(job,action),volume.commit,volume.reload)

@app.function(image=image,volumes={'/data':volume},schedule=modal.Period(hours=6),timeout=300)
def cleanup():
    import shutil
    volume.reload();now=time.time()
    for key,value in list(records.items()):
        if key.startswith('lock:'):
            if value.get('expires',0)<now:records.pop(key,None)
        elif isinstance(value,dict) and 'created' in value and now-value['created']>86400 and value.get('status') in ('awaiting_worker','review','completed','failed','cancelled'):
            shutil.rmtree(Path('/data')/key,ignore_errors=True);records.pop(key,None);records.pop('cancel:'+key,None)
    # Also remove orphan files after Dict expiry or interrupted uploads.
    for folder in Path('/data').iterdir():
        if folder.is_dir() and now-folder.stat().st_mtime>172800 and not records.contains(folder.name):shutil.rmtree(folder,ignore_errors=True)
    volume.commit()
