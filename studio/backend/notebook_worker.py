"""Run one user-started transcription task. No tunnel or unattended Colab worker."""
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import httpx
from backend.core import transcribe_audio

def run_once(execution,endpoint,token,workspace_id):
    if execution not in ('colab','kaggle','lightning'):raise ValueError('Choose colab, kaggle or lightning')
    parsed=urlparse(endpoint)
    if parsed.scheme!='https' or not parsed.hostname or not parsed.hostname.endswith('.modal.run'):raise ValueError('Use the deployed Modal HTTPS endpoint')
    if len(workspace_id)!=64 or any(c not in '0123456789abcdef' for c in workspace_id):raise ValueError('Copy the workspace ID from Connections')
    headers={'Authorization':'Bearer '+token,'X-Studio-Owner':workspace_id}
    with httpx.Client(base_url=endpoint.rstrip('/'),headers=headers,timeout=120,follow_redirects=False) as client:
        response=client.get('/worker/next',params={'execution':execution});response.raise_for_status();job=response.json()['job']
        if not job:
            print('No projects waiting for this worker. Choose this backend in the studio and upload a video first.');return
        print('Transcribing:',job['name'])
        with tempfile.TemporaryDirectory() as folder:
            audio=Path(folder)/'audio.wav';total=0
            with client.stream('GET',f'/worker/{job["id"]}/audio') as result,audio.open('wb') as output:
                result.raise_for_status()
                for chunk in result.iter_bytes(1024*1024):
                    total+=len(chunk)
                    if total>300_000_000:raise ValueError('Audio is too large')
                    output.write(chunk)
            result=transcribe_audio(audio,job['options']['source'],job['duration'])
            response=client.post(f'/worker/{job["id"]}/transcript',json=result);response.raise_for_status()
        print('Transcript returned. Translation and subtitle review continue in your studio.')
if __name__=='__main__':
    run_once(os.environ['STUDIO_EXECUTION'],os.environ['STUDIO_ENDPOINT'],os.environ['STUDIO_TOKEN'],os.environ['STUDIO_WORKSPACE'])
