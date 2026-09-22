"""Authenticated API, shared by Modal and a dedicated Lightning/local worker."""
import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from backend.core import Options, PipelineError, MAX_BYTES, readiness, probe, validate_segments

ARTIFACTS={'subtitles.srt':'application/x-subrip','subtitles.vtt':'text/vtt','segments.json':'application/json','output.mp4':'video/mp4'}
TERMINAL={'awaiting_worker','review','completed','failed','cancelled'}

def make_app(store, root, dispatch, commit=lambda:None, reload=lambda:None):
    api=FastAPI(title='VideoTrans Studio',docs_url=None,redoc_url=None,openapi_url=None)
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def auth(request:Request):
        token=os.getenv('STUDIO_TOKEN','')
        if len(token)<32:raise HTTPException(503,'Configure STUDIO_TOKEN with at least 32 characters.')
        supplied=request.headers.get('Authorization','').removeprefix('Bearer ')
        if not hmac.compare_digest(supplied,token):raise HTTPException(401,'Unauthorized')
        owner=request.headers.get('X-Studio-Owner','')
        if not re.fullmatch(r'[a-f0-9]{64}',owner):raise HTTPException(400,'Invalid workspace identity')
        return owner
    def owned(job_id,owner):
        if not re.fullmatch(r'[a-f0-9]{32}',job_id):raise HTTPException(404,'Project not found')
        job=store.get(job_id)
        if not job or job.get('owner')!=owner:raise HTTPException(404,'Project not found')
        return job
    def public(job):return {k:v for k,v in job.items() if k not in ('owner','call_id')}
    @api.exception_handler(PipelineError)
    async def pipeline_error(request,exc):return JSONResponse({'error':str(exc)},status_code=422)
    @api.get('/health')
    def health(owner=Depends(auth)):return readiness()
    @api.get('/jobs')
    def jobs(owner=Depends(auth)):
        rows=[public(v) for k,v in store.items() if isinstance(v,dict) and v.get('owner')==owner and not k.startswith('lock:')]
        return {'jobs':sorted(rows,key=lambda x:x['created'],reverse=True)[:100]}
    @api.post('/jobs')
    async def upload(request:Request,owner=Depends(auth)):
        try:
            name=base64.urlsafe_b64decode(request.headers['X-Video-Name']+'===').decode('utf-8')
            options=Options.model_validate_json(base64.urlsafe_b64decode(request.headers['X-Video-Options']+'==='))
            job_id=request.headers['X-Request-Id']
            if not re.fullmatch(r'[a-f0-9]{32}',job_id):raise ValueError()
            if len(name)>240 or not re.search(r'\.(mp4|mov|mkv|webm)$',name,re.I):raise ValueError()
        except Exception:raise HTTPException(400,'Invalid upload name, options or request identifier')
        existing=store.get(job_id)
        if existing:
            if existing.get('owner')!=owner:raise HTTPException(409,'Request identifier is in use')
            return {'job':public(existing)}
        if sum(1 for _,v in store.items() if isinstance(v,dict) and v.get('owner')==owner and v.get('status') not in TERMINAL)>=3:
            raise HTTPException(429,'Three projects are already processing. Wait for one to finish.')
        health=readiness()
        if not health['providers'][options.provider]:raise HTTPException(503,'Configure the selected translation provider first.')
        if options.separate and not health['mvsep']:raise HTTPException(503,'Configure MVSEP or disable speech isolation.')
        # Atomic distributed lock prevents duplicate upload/dispatch for retries.
        if not store.lock(job_id):raise HTTPException(409,'This upload is already in progress.')
        folder=root/job_id
        try:
            await asyncio.to_thread(reload)
            folder.mkdir(exist_ok=True)
            total=0
            async with asyncio.timeout(300):
                with (folder/'input.video').open('wb') as f:
                    async for chunk in request.stream():
                        total+=len(chunk)
                        if total>=MAX_BYTES:raise HTTPException(413,'Video must be smaller than 100 MB.')
                        f.write(chunk)
            duration=await asyncio.to_thread(probe,folder/'input.video')
            job={'id':job_id,'owner':owner,'name':name,'status':'queued','progress':0,'created':time.time(),'updated':time.time(),'options':options.model_dump(),'duration':duration,'revision':0}
            store.set(job_id,job)
            await asyncio.to_thread(commit)
            try:await asyncio.to_thread(dispatch,job_id,'process')
            except Exception:
                job.update(status='failed',error='Could not dispatch processing. Try again.',updated=time.time());store.set(job_id,job)
            return {'job':public(job)}
        except BaseException:
            if not store.get(job_id):shutil.rmtree(folder,ignore_errors=True)
            raise
        finally:store.unlock(job_id)
    @api.get('/jobs/{job_id}')
    def detail(job_id:str,owner=Depends(auth)):
        job=owned(job_id,owner)
        result=public(job)
        if job['status'] in ('review','completed'):
            reload();path=root/job_id/'segments.json'
            if path.exists():result['segments']=json.loads(path.read_text())
        if job['status'] not in TERMINAL and time.time()-job['updated']>3900:
            job.update(status='failed',error='Processing stopped or timed out. Retry this project.',updated=time.time());store.set(job_id,job);result=public(job)
        return {'job':result}
    class Edit(BaseModel):
        revision:int=Field(ge=0)
        segments:list=Field(min_length=1,max_length=3000)
    @api.patch('/jobs/{job_id}')
    def edit(job_id:str,body:Edit,owner=Depends(auth)):
        if not store.lock(job_id):raise HTTPException(409,'Project is busy')
        try:
            job=owned(job_id,owner)
            if job['status'] not in ('review','completed'):raise HTTPException(409,'Project is not ready for editing')
            if job['revision']!=body.revision:raise HTTPException(409,'This project changed. Reload before saving.')
            segments=validate_segments(body.segments)
            if any(s['end']>job['duration']+.05 for s in segments):raise HTTPException(422,'Subtitle times exceed video duration')
            reload();folder=root/job_id
            from backend.core import export_files
            export_files(folder,segments,False)
            (folder/'output.mp4').unlink(missing_ok=True)
            commit();job.update(status='review',progress=95,revision=job['revision']+1,updated=time.time());store.set(job_id,job)
            return {'job':{**public(job),'segments':segments}}
        finally:store.unlock(job_id)
    @api.post('/jobs/{job_id}/render')
    def render(job_id:str,owner=Depends(auth)):
        if not store.lock(job_id):raise HTTPException(409,'Project is busy')
        try:
            job=owned(job_id,owner)
            if job['status']=='rendering':return {'job':public(job)}
            if job['status'] not in ('review','completed'):raise HTTPException(409,'Review subtitles before exporting')
            job.update(status='rendering',progress=96,updated=time.time());store.set(job_id,job)
            try:dispatch(job_id,'render')
            except Exception:
                job.update(status='failed',error='Could not dispatch export. Retry the project.');store.set(job_id,job)
            return {'job':public(job)}
        finally:store.unlock(job_id)
    @api.post('/jobs/{job_id}/retry')
    def retry(job_id:str,owner=Depends(auth)):
        if not store.lock(job_id):raise HTTPException(409,'Project is busy')
        try:
            job=owned(job_id,owner)
            if job['status'] not in ('failed','cancelled'):raise HTTPException(409,'Only failed or cancelled projects can be retried')
            store.delete('cancel:'+job_id);job.pop('error',None)
            job.update(status='queued',progress=0,updated=time.time());store.set(job_id,job)
            try:dispatch(job_id,'process')
            except Exception:
                job.update(status='failed',error='Could not dispatch retry.');store.set(job_id,job)
            return {'job':public(job)}
        finally:store.unlock(job_id)
    @api.post('/jobs/{job_id}/cancel')
    def cancel(job_id:str,owner=Depends(auth)):
        job=owned(job_id,owner)
        if job['status']=='awaiting_worker':
            job.update(status='cancelled',updated=time.time());store.set(job_id,job)
        elif job['status'] not in TERMINAL:
            store.set('cancel:'+job_id,True)
        return {'accepted':True}
    @api.get('/jobs/{job_id}/files/{name}')
    def download(job_id:str,name:str,owner=Depends(auth)):
        job=owned(job_id,owner)
        if job['status'] not in ('review','completed') or name not in ARTIFACTS:raise HTTPException(404,'Export not ready')
        reload();path=root/job_id/name
        if not path.is_file():raise HTTPException(404,'Export not ready')
        return FileResponse(path,media_type=ARTIFACTS[name],filename=name,headers={'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'})
    @api.delete('/jobs/{job_id}')
    def delete(job_id:str,owner=Depends(auth)):
        if not store.lock(job_id):raise HTTPException(409,'Project is busy')
        try:
            job=owned(job_id,owner)
            if job['status'] not in TERMINAL:raise HTTPException(409,'Cancel processing and wait before deleting')
            reload();shutil.rmtree(root/job_id,ignore_errors=True);commit();store.delete(job_id);store.delete('cancel:'+job_id)
            return {'deleted':True}
        finally:store.unlock(job_id)
    @api.get('/worker/next')
    def worker_next(execution:str,owner=Depends(auth)):
        rows=[v for _,v in store.items() if isinstance(v,dict) and v.get('owner')==owner and v.get('status')=='awaiting_worker' and v.get('options',{}).get('execution')==execution]
        rows.sort(key=lambda x:x['created'])
        return {'job':public(rows[0]) if rows else None}
    @api.get('/worker/{job_id}/audio')
    def worker_audio(job_id:str,owner=Depends(auth)):
        job=owned(job_id,owner)
        if job['status']!='awaiting_worker':raise HTTPException(409,'Project is not awaiting a worker')
        reload();return FileResponse(root/job_id/'audio.wav',media_type='audio/wav')
    class Transcript(BaseModel):
        segments:list=Field(min_length=1,max_length=3000)
        language:str=Field(min_length=2,max_length=10)
    @api.post('/worker/{job_id}/transcript')
    def worker_result(job_id:str,body:Transcript,owner=Depends(auth)):
        if not store.lock(job_id):raise HTTPException(409,'Project is busy')
        try:
            job=owned(job_id,owner)
            if job['status']!='awaiting_worker':raise HTTPException(409,'Project no longer accepts a transcript')
            segments=validate_segments(body.segments)
            if any(s['end']>job['duration']+.05 for s in segments):raise HTTPException(422,'Subtitle times exceed video duration')
            reload();(root/job_id/'transcript.json').write_text(json.dumps({'segments':segments,'language':body.language},ensure_ascii=False))
            commit();job.update(status='queued',progress=44,updated=time.time());store.set(job_id,job)
            try:dispatch(job_id,'process')
            except Exception:
                job.update(status='failed',error='Could not dispatch translation.');store.set(job_id,job)
            return {'accepted':True}
        finally:store.unlock(job_id)
    return api


def execute(store,root,job_id,action,commit=lambda:None,reload=lambda:None):
    from backend.core import process,export_files
    job=store.get(job_id)
    if not job:return
    def cancelled():
        if store.get('cancel:'+job_id):raise PipelineError('Processing cancelled.')
    def progress(status,amount):
        job.update(status=status,progress=amount,updated=time.time());store.set(job_id,job)
    try:
        reload();cancelled();folder=Path(root)/job_id
        if action=='process':
            result=process(folder,Options.model_validate(job['options']),progress,cancelled)
            cancelled();commit();progress('awaiting_worker',15) if result is None else progress('review',95)
        else:
            segments=json.loads((folder/'segments.json').read_text())
            export_files(folder,segments,job['options']['burn'])
            cancelled();commit();progress('completed',100)
    except Exception as exc:
        is_cancelled=bool(store.get('cancel:'+job_id))
        job.update(status='cancelled' if is_cancelled else 'failed',error=str(exc) if isinstance(exc,PipelineError) else 'Processing failed. Check the worker logs and provider configuration.',updated=time.time())
        store.set(job_id,job)
