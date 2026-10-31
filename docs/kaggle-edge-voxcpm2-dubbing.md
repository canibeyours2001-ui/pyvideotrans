# Kaggle pyVideoTrans: Microsoft Edge TTS or VoxCPM2

**Status: feature branch and offline checks. This is NOT a live Telegram API endpoint.**
The original VoxCPM2 Kaggle notebook and the existing Cloudflare Telegram Worker have not been changed.

## Design

Use pyVideoTrans for:
1. Speech recognition from the input video, if doing full video translation (`--task vtv`).
2. Subtitle translation and timing/alignment.
3. **Choose dubbing provider at job submission**:
   - `edge` = **Microsoft Edge TTS (provider ID 0)**. Online speech synthesis; no GPU required *for TTS*, but `vtv` can still need GPU for Whisper.
   - `voxcpm2` = **native VoxCPM2 Kaggle GPU (provider ID 38)**. Loads `openbmb/VoxCPM2` only once, generates each subtitle segment sequentially using the model already proven in the user's separate VoxCPM2 Kaggle setup. No separate Gradio HTTP endpoint.
4. Align audio with original subtitle timestamps, and produce a dubbed video or WAV using the existing CLI.

The existing VoxCPM Gradio API provider (ID 15) is unchanged and unrelated.

## Kaggle input and output

- Inputs must first exist **inside the Kaggle notebook filesystem**, such as a *private Kaggle Dataset* containing the MP4/SRT and an optional 5–10 second authorized reference WAV.
- For a source video, pyVideoTrans's CLI performs STT → translation → dubbing → alignment → assembly (`--task vtv`).
- If a translated SRT already exists, run `--task tts` to generate timed dubbing directly (often faster for already-translated videos).
- Intermediate files, models, and caches belong in `/kaggle/temp`; the launcher copies the final WAV/MP4 and a small JSON status file into `/kaggle/working` so saved notebook outputs remain small.
- To make this an **automatic Telegram upload → private Kaggle batch → delivered MP4** pipeline, add secure video staging and a new Cloudflare D1 queue. The existing VoxCPM2 text-only queue has no video upload or processing pipeline; DO NOT reuse its 8,000-character job schema for videos.

## Commands inside a compatible Python 3.10 pyVideoTrans environment

Choose Edge TTS (subtitles already translated):

```bash
python tools/kaggle_video_dub_job.py \
  --task tts --input /kaggle/input/subtitles/translated.srt \
  --tts edge --voice-role en-US-AriaNeural \
  --target-language-code en
```

Choose VoxCPM2 on T4 GPU, with a short consented voice reference:

```bash
VOXCPM_MODEL_DIR=/kaggle/temp/VoxCPM2 \
python tools/kaggle_video_dub_job.py \
  --task tts --input /kaggle/input/subtitles/translated.srt \
  --tts voxcpm2 --voice-role Narrator --cuda \
  --target-language-code en \
  --reference-wav /kaggle/input/my-voice/reference.wav
```

Full video dubbing (only after source language, target language and translator are verified):

```bash
python tools/kaggle_video_dub_job.py \
  --task vtv --input /kaggle/input/videos/clip.mp4 \
  --tts edge --voice-role en-US-AriaNeural \
  --source-language-code zh-cn --target-language-code en \
  --recogn-type 0 --whisper-model small --translate-type 0 \
  --voice-autorate
```

Switch `--tts edge` to `--tts voxcpm2 --voice-role Narrator --cuda` to use Kaggle GPU dubbing. Translation quality and language support depend on the configured pyVideoTrans provider; validate Burmese `my` in the installed app before claiming support for a particular translation engine.

## Environment constraints (must test on Kaggle)

pyVideoTrans's own `pyproject.toml` and official setup docs specify **Python 3.10** (`>=3.10,<3.11`). Kaggle's current default Python image can differ. Use a Python 3.10 environment and install pyVideoTrans's real dependencies (including FFmpeg and CUDA for VoxCPM2) before launching the script. Full `uv sync` may take time and substantial storage.

- VoxCPM2 model downloads into `/kaggle/temp/VoxCPM2` to avoid bloating notebook outputs.
- Microsoft Edge TTS is not an officially guaranteed free unlimited production API. It depends on network availability and can be rate-limited.
- Preserve private notebook and datasets. Do not embed the bot token, Kaggle token, or private customer video data into public kernel source.
- Kaggle only runs during a queued notebook execution; it is **not** an always-on HTTP inference server.

## Validation

```bash
python tools/test_kaggle_dual_tts.py
python -m py_compile videotrans/tts/_voxcpm2kaggle.py tools/kaggle_video_dub_job.py
```

These checks test registration, CLI provider selection and syntax **without consuming GPU quota**. A real Kaggle video + MP3/WAV/MP4 end-to-end test is still required.

References:
- https://en.pyvideotrans.com/cli
- https://en.pyvideotrans.com/faq
- https://www.kaggle.com/docs/notebooks
