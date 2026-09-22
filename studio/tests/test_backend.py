import base64,json,os,subprocess,time
from pathlib import Path
from unittest.mock import patch
import pytest
import httpx
from fastapi.testclient import TestClient
from backend.core import Options,PipelineError,probe,subtitle_text,validate_segments,translate,export_files
from backend.service import make_app,execute

class MemoryStore:
    def __init__(self):self.data={}
    def get(self,k):return self.data.get(k)
    def set(self,k,v):self.data[k]=dict(v) if isinstance(v,dict) else v
    def items(self):return list(self.data.items())
    def delete(self,k):self.data.pop(k,None)
    def lock(self,k):
        if 'lock:'+k in self.data:return False
        self.data['lock:'+k]=True;return True
    def unlock(self,k):self.delete('lock:'+k)

@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setenv('STUDIO_TOKEN','x'*40)
    for k,v in {'OMNIROUTE_BASE_URL':'https://translate.example/v1','OMNIROUTE_API_KEY':'test-secret','OMNIROUTE_MODEL':'test-model'}.items():monkeypatch.setenv(k,v)
    store=MemoryStore();calls=[]
    client=TestClient(make_app(store,tmp_path,lambda *v:calls.append(v)))
    headers={'Authorization':'Bearer '+'x'*40,'X-Studio-Owner':'a'*64}
    return client,store,headers,calls,tmp_path

@pytest.fixture
def media(tmp_path):
    path=tmp_path/'fixture.mp4'
    subprocess.run(['ffmpeg','-y','-v','error','-f','lavfi','-i','color=c=blue:s=320x240:d=2','-f','lavfi','-i','sine=frequency=440:duration=2','-c:v','libx264','-c:a','aac','-shortest',str(path)],check=True)
    return path

def upload_headers(headers,job='1'*32):
    enc=lambda s:base64.urlsafe_b64encode(s.encode()).decode().rstrip('=')
    return {**headers,'X-Video-Name':enc('မြန်မာ.mp4'),'X-Video-Options':enc(json.dumps({'burn':False})),'X-Request-Id':job}

def test_auth_required(setup):
    c,*_=setup
    assert c.get('/jobs').status_code==401
    assert c.get('/health').status_code==401

def test_upload_and_idempotency(setup,media):
    c,s,h,calls,root=setup
    r=c.post('/jobs',content=media.read_bytes(),headers=upload_headers(h));assert r.status_code==200
    assert r.json()['job']['name']=='မြန်မာ.mp4'
    assert 'owner' not in r.json()['job']
    assert c.post('/jobs',content=b'',headers=upload_headers(h)).status_code==200
    assert len(calls)==1

def test_owner_isolation(setup,media):
    c,s,h,_,_=setup;c.post('/jobs',content=media.read_bytes(),headers=upload_headers(h))
    other={**h,'X-Studio-Owner':'b'*64}
    assert c.get('/jobs/'+'1'*32,headers=other).status_code==404
    assert c.get('/jobs',headers=other).json()['jobs']==[]
    assert c.post('/jobs',content=media.read_bytes(),headers=upload_headers(other)).status_code==409

def test_invalid_video_removed(setup):
    c,s,h,_,root=setup
    assert c.post('/jobs',content=b'fake video',headers=upload_headers(h)).status_code==422
    assert not (root/('1'*32)).exists()

def test_size_limit_streamed(setup,monkeypatch):
    c,s,h,_,root=setup;monkeypatch.setattr('backend.service.MAX_BYTES',10)
    assert c.post('/jobs',content=b'x'*10,headers=upload_headers(h)).status_code==413
    assert not (root/('1'*32)).exists()

def test_duration_boundary(tmp_path):
    p=tmp_path/'input';p.write_bytes(b'1')
    for value in ['600','nan','inf','0','-1']:
        with patch('backend.core.run',return_value=json.dumps({'format':{'duration':value},'streams':[{'codec_type':'video'},{'codec_type':'audio'}]}).encode()),pytest.raises(PipelineError):probe(p)

def test_valid_media_probe(media):assert 1.9<probe(media)<2.2

def test_invalid_times():
    for a,b in [(2,1),(float('nan'),2),(0,600)]:
        with pytest.raises(Exception):validate_segments([{'start':a,'end':b,'text':'a','translation':'b'}])

def test_unicode_subtitle_and_injection():
    seg=[{'start':0,'end':1.125,'text':'Hello','translation':'မင်္ဂလာပါ\n\n<b>hello</b> -->'}]
    text=subtitle_text(seg)
    assert 'မင်္ဂလာပါ' in text and '00:00:01,125' in text and '<b>' not in text
    assert text.count('-->')==1
    assert subtitle_text(seg,True).startswith('WEBVTT\n\n')

def test_translation_preserves_order(setup):
    client,store,h,calls,root=setup
    rows=[{'start':0,'end':1,'text':'Hello','translation':''},{'start':1,'end':2,'text':'World','translation':''}]
    transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'choices':[{'message':{'content':json.dumps([{'id':1,'translation':'ကမ္ဘာ'},{'id':0,'translation':'မင်္ဂလာပါ'}])}}]}))
    with patch('backend.core.httpx.Client',return_value=httpx.Client(transport=transport)):
        result=translate(rows,Options(),lambda *a:None,lambda:None)
    assert result[0]['translation']=='မင်္ဂလာပါ' and result[1]['start']==1

def test_translation_rejects_missing_rows(setup):
    transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'choices':[{'message':{'content':'[]'}}]}))
    with patch('backend.core.httpx.Client',return_value=httpx.Client(transport=transport)),patch('backend.core.time.sleep'),pytest.raises(PipelineError):
        translate([{'start':0,'end':1,'text':'Hi','translation':''}],Options(),lambda *a:None,lambda:None)

def test_provider_key_not_exposed(setup):
    c,s,h,_,_=setup;r=c.get('/health',headers=h)
    assert r.status_code==200 and 'test-secret' not in r.text and r.json()['providers']['omniroute']

def test_review_edit_export_lifecycle(setup,media):
    c,s,h,_,root=setup;c.post('/jobs',content=media.read_bytes(),headers=upload_headers(h))
    segments=[{'start':0,'end':1,'text':'Hello','translation':'မင်္ဂလာပါ'}]
    def fake_process(folder,*args):export_files(folder,segments,False);return segments
    with patch('backend.core.process',fake_process):execute(s,root,'1'*32,'process')
    r=c.get('/jobs/'+'1'*32,headers=h);assert r.json()['job']['status']=='review'
    assert r.json()['job']['segments']==segments
    assert c.patch('/jobs/'+'1'*32,headers=h,json={'revision':5,'segments':segments}).status_code==409
    assert c.patch('/jobs/'+'1'*32,headers=h,json={'revision':0,'segments':segments}).status_code==200
    assert c.get('/jobs/'+'1'*32+'/files/subtitles.srt',headers=h).status_code==200
    assert c.get('/jobs/'+'1'*32+'/files/output.mp4',headers=h).status_code==404
    assert c.post('/jobs/'+'1'*32+'/render',headers=h).status_code==200
    execute(s,root,'1'*32,'render')
    assert s.get('1'*32)['status']=='completed'
    assert c.delete('/jobs/'+'1'*32,headers=h).status_code==200

def test_cancel_before_processing(setup,media):
    c,s,h,_,root=setup;c.post('/jobs',content=media.read_bytes(),headers=upload_headers(h))
    c.post('/jobs/'+'1'*32+'/cancel',headers=h);execute(s,root,'1'*32,'process')
    assert s.get('1'*32)['status']=='cancelled'

def test_real_ffmpeg_export(tmp_path,media):
    import shutil
    if 'subtitles' not in subprocess.run(['ffmpeg','-filters'],capture_output=True,text=True).stdout:pytest.skip('FFmpeg libass filter not installed')
    folder=tmp_path/'render';folder.mkdir();shutil.copy(media,folder/'input.video')
    export_files(folder,[{'start':0,'end':1,'text':'Hello','translation':'Hello world'}],True)
    assert (folder/'output.mp4').stat().st_size>0

def test_external_worker_roundtrip_and_replay(setup,media):
    c,s,h,calls,root=setup
    headers=upload_headers(h)
    headers['X-Video-Options']=base64.urlsafe_b64encode(json.dumps({'execution':'kaggle','burn':False}).encode()).decode()
    c.post('/jobs',content=media.read_bytes(),headers=headers)
    execute(s,root,'1'*32,'process')
    assert s.get('1'*32)['status']=='awaiting_worker'
    assert c.get('/worker/next?execution=colab',headers=h).json()['job'] is None
    assert c.get('/worker/next?execution=kaggle',headers=h).json()['job']['id']=='1'*32
    assert c.get('/worker/'+'1'*32+'/audio',headers=h).status_code==200
    payload={'segments':[{'start':0,'end':1,'text':'မင်္ဂလာပါ','translation':'မင်္ဂလာပါ'}],'language':'my'}
    assert c.post('/worker/'+'1'*32+'/transcript',headers=h,json=payload).status_code==200
    assert c.post('/worker/'+'1'*32+'/transcript',headers=h,json=payload).status_code==409
    execute(s,root,'1'*32,'process')
    assert s.get('1'*32)['status']=='review'

def test_failed_job_retry(setup,media):
    c,s,h,calls,root=setup;c.post('/jobs',content=media.read_bytes(),headers=upload_headers(h))
    job=s.get('1'*32);job.update(status='failed',error='An error');s.set('1'*32,job)
    assert c.post('/jobs/'+'1'*32+'/retry',headers=h).status_code==200
    assert s.get('1'*32)['status']=='queued' and 'error' not in s.get('1'*32)
    assert len(calls)==2

def test_mvsep_result_contract(setup,monkeypatch,tmp_path):
    from backend.core import separate_audio
    monkeypatch.setenv('MVSEP_API_KEY','mvsep-test');audio=tmp_path/'audio.wav';audio.write_bytes(b'input')
    def handler(request):
        if request.url.path.endswith('/create'):return httpx.Response(200,json={'success':True,'data':{'hash':'abc'}})
        if request.url.path.endswith('/get'):return httpx.Response(200,json={'success':True,'status':'done','data':{'files':[{'download':'track_vocals.wav','url':'https://sg.mvsep.com/results/v.wav'}]}})
        return httpx.Response(200,content=b'vocals')
    with patch('backend.core.httpx.Client',return_value=httpx.Client(transport=httpx.MockTransport(handler))):
        assert separate_audio(audio,lambda:None).read_bytes()==b'vocals'
