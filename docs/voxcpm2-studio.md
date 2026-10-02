# VoxCPM2 Studio integration

This branch adds VoxCPM2 as a **built-in pyVideoTrans TTS provider** rather than the existing external VoxCPM Gradio-API channel.

## Main video workflow

Choose **VoxCPM2 (Built-in)** in the TTS selector. The normal pyVideoTrans subtitle/dubbing/alignment pipeline remains unchanged, but all subtitle TTS segments are sent to a bounded VoxCPM2 scheduler:

- two CUDA GPUs: one model replica per GPU, up to two subtitle segments generated in parallel;
- one CUDA GPU: one worker automatically;
- no CUDA: CPU fallback (slow);
- next segment is dispatched as soon as a worker finishes;
- speed is applied afterwards with FFmpeg `atempo`;
- Burmese text is sent with VoxCPM normalization disabled.

The model defaults to `openbmb/VoxCPM2`. Set `VOXCPM2_MODEL_DIR=/path/to/VoxCPM2` to use pre-downloaded weights.

## VoxCPM2 Voice Studio

Selecting the built-in channel reveals:

- Live Voiceover with Gradio streaming audio, autoplay request, stop/cancel, speed, CFG, diffusion steps, chunk length and join pause.
- Voice Design with 12 preset profiles and candidate audition/save.
- Voice Clone from local audio/video or remote media.
- Automatic Burmese reference transcription via `freococo/myanmar_asr`; other listed languages use faster-whisper large-v3.
- Persistent voice library with saved cloned/designed voices and delete.
- Delivery presets and custom tone/emotion control.

Saved voices live in `models/voxcpm2/voices/` and are exposed in the normal pyVideoTrans voice-role menu.

## Remote video input

The WebUI has two source modes:

- **Upload file**
- **Remote upload**

The URL/cookie/header controls are only visible for Remote upload. yt-dlp is used for the broad platform set it supports, with explicit Google Drive and MEGA adapters and a direct HTTP fallback.

Only download media you have permission to access. Site authentication/cookies remain subject to the source platform's policies.

## Video Editing Room

The post-production tab can load the latest pyVideoTrans result and:

- blur the bottom subtitle area of an existing video;
- burn the generated SRT/ASS/VTT subtitles;
- overlay a transparent image/logo;
- control logo opacity, width and corner;
- render/download an H.264/AAC MP4.

## Dependencies

The branch adds:

- `voxcpm>=2.0.3`
- `yt-dlp`
- `gdown`
- `mega-py-v2`

VoxCPM2 itself requires an environment compatible with its PyTorch/CUDA requirements.
