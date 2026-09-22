# VideoTrans Studio

Private English/Burmese web studio for the pyvideotrans fork. Upload a video **strictly under 100,000,000 bytes and 600 seconds**, transcribe with Whisper large-v3, translate through OmniRoute or FreeLLMAPI, review/edit subtitles, and export SRT, VTT, JSON and an optional subtitled MP4.

This is a separate web module using the same faster-whisper transcription engine as the fork's existing Kaggle worker. It does not import the desktop GUI or reproduce all pyVideoTrans features. Voice dubbing and OCR are not implemented. Existing desktop and hybrid workflows remain available.

## Credential setup

1. Add repository secrets at https://github.com/canibeyours2001-ui/pyvideotrans/settings/secrets/actions:
   - `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`: Modal deployment credentials.
   - `STUDIO_TOKEN`: a random value with at least 32 characters; keep it private.
   - For at least one translation provider, set all three corresponding values:
     - `OMNIROUTE_API_KEY`, `OMNIROUTE_BASE_URL`, `OMNIROUTE_MODEL`
     - `FREELLMAPI_API_KEY`, `FREELLMAPI_BASE_URL`, `FREELLMAPI_MODEL`
   - Optional: `MVSEP_API_KEY`, `MVSEP_SEP_TYPE` (defaults to vocals model 40).
2. Base URLs must be HTTPS, ending at the OpenAI-compatible API root, usually `/v1`. The app appends `/chat/completions`. Supply the exact model identifier offered by your service. “FreeLLMAPI” is intentionally configurable because no specific vendor endpoint was supplied.
3. Run **Deploy VideoTrans Studio** in GitHub Actions after this change is on the default branch. It securely synchronizes these named secrets to Modal and deploys the worker. Redeploy after changing secrets.
4. In the private website's **Connections** tab, paste the resulting `https://…modal.run` endpoint and the same `STUDIO_TOKEN`. The endpoint is verified; the token is AES-GCM encrypted in D1. The website's separate `SETTINGS_ENCRYPTION_KEY` is stored as a hosting secret.
5. Upload a short test clip. Wait for review, check the translation, approve export and download the result. Real service tests are required before production acceptance; passing unit tests alone does not verify provider credentials, quotas or GPU deployment.

GitHub secrets are write-only. Existing `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN` and `RCLONE_MEGA_CONFIG` references belong to the fork's older hybrid workflow; their values are not extracted or copied into this studio.

## Execution backends

| Service | Role |
| --- | --- |
| Modal | Always-on-demand coordinator, durable files/job state, MVSEP, translation and export; automatic T4 transcription by default |
| Colab | User-started notebook runs one queued transcription and returns its result |
| Kaggle | User-started GPU notebook runs one queued transcription and returns its result |
| Lightning AI | Same single-job adapter in a GPU Studio; optional single-host API/queue deployment |
| GitHub Actions | CI and explicit deployment; not a video-processing server |
| MVSEP | Optional vocal isolation before transcription |

For notebooks, choose the corresponding processing backend in the website before upload. Once the job says **Waiting for notebook**, open `notebooks/colab.ipynb`, `kaggle.ipynb`, or `lightning.ipynb`. Add `STUDIO_ENDPOINT`, `STUDIO_TOKEN`, and `STUDIO_WORKSPACE` in that platform's secret store. The workspace ID is displayed under Connections. Enable GPU and internet, then run the notebook once. It downloads that project's audio, runs large-v3 and submits the transcript. The website then resumes translation. No tunnel or unattended free-Colab worker is used.

Colab and Kaggle are interactive sessions, with platform quotas and possible interruptions; they are not transparent automatic failover. Modal is still required as the coordinator in the hosted configuration. Provider/GPU/storage charges may apply. No provider is assumed to be free or unlimited.

## Data, security and operations

- New website is owner-private. APIs require the hosting platform's authenticated identity.
- Website only connects to HTTPS `*.modal.run` endpoints. Backend requires a constant-time bearer-token comparison and a workspace identity. Every project lookup/download enforces ownership.
- Upload bodies stream through the website; they are never buffered as a 100 MB browser multipart object on a 128 MB edge isolate. The worker enforces the actual byte limit and probes actual media duration, audio/video tracks and maximum 4K resolution.
- No arbitrary media URLs, shell interpolation, client-supplied file paths, public artifact URLs, or browser-side provider keys.
- At most three active projects per workspace, and one GPU worker container. Translation retries are bounded; malformed/omitted LLM rows cannot silently change subtitle alignment.
- Modal files are stored on a Volume; metadata uses a Dict. Finished/failed/cancelled/waiting projects expire after 24 hours, cleaned every six hours (24–30 hours total). Downloads should be saved before expiry. Dict objects have a platform inactivity expiry; periodic cleanup also removes orphaned files.
- API secrets remain in Modal Secrets. The deployment step intentionally overwrites only the explicit studio secret keys, including blank values when a provider is removed.
- MVSEP receives audio when isolation is enabled. The chosen translation service receives subtitle text. The selected notebook platform receives audio. Retention at those external services follows their own policies.
- Cancellation is cooperative between processing stages; a running FFmpeg or provider request may finish before cancellation takes effect. A cancelled job can be retried.
- Failed workers expose a generic error, not credentials or provider response bodies. Check private Modal logs for infrastructure issues. Stale active jobs are surfaced as failed after the configured timeout.
- This deployment is intended for one owner/private workspace. Public multi-tenant SaaS, billing, stronger quota controls and externally managed identities require additional work.

## Development

Frontend: Node 22.13+; `npm run install:ci`, `npm run dev`. The committed hosting manifest declares the D1 connection table. Production migrations are in `drizzle/` and are applied by Sites hosting.

Backend tests (Python 3.11+, FFmpeg):

```sh
python -m pip install fastapi==0.115.12 httpx==0.28.1 pydantic==2.11.4 pytest
python -m pytest tests -q
```

Local/Lightning standalone service: install `backend/requirements.txt`, set backend environment variables, then run `uvicorn backend.local:api --host 0.0.0.0 --port 8000 --workers 1`. Use exactly one process for its durable SQLite queue. The hosted website intentionally only accepts Modal endpoints; the standalone service is for a separately secured deployment. The Dockerfile includes CUDA runtime libraries; use an NVIDIA-capable host and GPU container runtime.

## Acceptance still required with credentials

- Deploy the actual Modal image and confirm model download/GPU compatibility.
- Process a real English-to-Burmese clip and a Burmese-to-English clip through each chosen translation provider.
- Run MVSEP with the user's account and selected separation model.
- Run one notebook job on each platform the user intends to use.
- Confirm live upload, review/edit, render, download, cancellation, retry and retention behavior.
- Have a fluent Burmese speaker review translated content and rendered glyph layout.

Tests use synthetic media and mocked speech/provider responses where credentials or a GPU are required. They do not claim real model or provider verification.

License: GPL-3.0, consistent with the parent pyvideotrans repository; third-party components retain their own licenses.
