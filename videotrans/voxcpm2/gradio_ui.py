from __future__ import annotations

from pathlib import Path

from videotrans.voxcpm2 import gradio_tools as tools
from videotrans.webui_media import download_remote_media, latest_output, render_edit_room


DELIVERY = [
    "Natural / keep cloned delivery",
    "Warm & conversational",
    "Calm & serious",
    "Friendly",
    "Professional / formal",
    "Documentary",
    "Movie recap",
    "Excited",
    "Soft / emotional",
]


def delivery_instruction(value):
    return {
        "Natural / keep cloned delivery":"",
        "Warm & conversational":"warm, natural, conversational delivery with subtle human pauses",
        "Calm & serious":"calm, serious, controlled delivery with clear articulation",
        "Friendly":"friendly, approachable, relaxed conversational delivery",
        "Professional / formal":"professional, formal, polished delivery with precise articulation",
        "Documentary":"deep, restrained documentary narration with subtle suspense",
        "Movie recap":"fast-paced engaging movie recap delivery with strong momentum and clean articulation",
        "Excited":"energetic and excited but controlled, clear and never shouting",
        "Soft / emotional":"soft, intimate and sincere delivery with gentle emotional variation",
    }.get(value,"")


def build_voxcpm2_studio(gr, visible=False):
    """Build the VoxCPM2-only controls inside pyVideoTrans WebUI."""
    with gr.Column(visible=visible) as panel:
        gr.Markdown("## 🎙 VoxCPM2 Voice Studio\nBuilt-in dual-GPU TTS • voice design • cloning • live streaming • persistent voice library")

        with gr.Tab("⚡ Live Voiceover"):
            live_role=gr.Dropdown(choices=tools.role_choices(),value="No",label="Voice / profile")
            live_script=gr.Textbox(lines=8,label="Script")
            with gr.Row():
                delivery=gr.Dropdown(choices=DELIVERY,value=DELIVERY[0],label="Delivery")
                custom=gr.Textbox(label="Custom tone / emotion",placeholder="calm, tense, cheerful…")
            with gr.Row():
                speed=gr.Slider(.6,1.8,1.0,.05,label="Generation speed")
                cfg=gr.Slider(1.0,4.0,2.0,.1,label="CFG")
                steps=gr.Slider(5,30,10,1,label="Inference steps")
            with gr.Row():
                chunk_chars=gr.Slider(80,450,230,10,label="Chunk characters")
                join_pause=gr.Slider(0,1000,300,25,label="Join pause (ms)")
            with gr.Row():
                live_btn=gr.Button("⚡ Generate + Live Stream",variant="primary")
                live_stop=gr.Button("■ Stop generating",variant="stop")
            live_audio=gr.Audio(label="Live combined preview",streaming=True,autoplay=True)
            live_status=gr.Markdown()
            live_file=gr.File(label="Final WAV")
            def _stream(script,role,delivery_name,custom_text,speed_value,cfg_value,step_value,chars,pause):
                instruction=delivery_instruction(delivery_name)
                for audio,status,final_file in tools.stream_voiceover(
                    script,role,instruction,custom_text,speed_value,cfg_value,step_value,chars,pause
                ):
                    yield (
                        gr.skip() if audio is None else audio,
                        status,
                        gr.skip() if final_file is None else final_file,
                    )
            live_event=live_btn.click(
                _stream,
                inputs=[live_script,live_role,delivery,custom,speed,cfg,steps,chunk_chars,join_pause],
                outputs=[live_audio,live_status,live_file],
            )
            live_stop.click(fn=None,cancels=[live_event])

        with gr.Tab("🎭 Voice Design"):
            gr.Markdown(
                "Choose a profile to generate an audition automatically. "
                "If you do not like the result, click **Try this profile again** "
                "or choose another profile. When you find a voice you like, "
                "click **Use this generated voice for video**."
            )
            profile=gr.Dropdown(
                choices=tools.profile_choices(),
                value=None,
                label="Voice profile",
                placeholder="Choose a profile to audition…",
            )
            design_text=gr.Textbox(
                lines=4,
                label="Audition text",
                value="မင်္ဂလာပါ။ ဒီအသံကို စမ်းသပ်နေပါတယ်။",
            )
            with gr.Row():
                design_btn=gr.Button("🎲 Try this profile again")
                use_candidate_btn=gr.Button(
                    "✅ Use this generated voice for video",
                    variant="primary",
                )
            design_audio=gr.Audio(
                label="Generated voice audition",
                autoplay=True,
            )
            candidate_state=gr.State("")
            design_status=gr.Markdown(
                "Select a profile above to generate the first audition."
            )

            # Selecting a different profile immediately generates its audition.
            profile.change(
                tools.generate_profile_candidate,
                inputs=[profile,design_text,cfg,steps],
                outputs=[design_audio,candidate_state,design_status],
            )

            # Explicit retry guarantees a fresh attempt even when the same
            # profile remains selected.
            design_btn.click(
                tools.generate_profile_candidate,
                inputs=[profile,design_text,cfg,steps],
                outputs=[design_audio,candidate_state,design_status],
            )

            with gr.Row():
                design_name=gr.Textbox(value="My Burmese Voice",label="Save name")
                save_design=gr.Button("💾 Save to voice library")
            save_design.click(
                lambda n,p,t: _save_role_result(gr,tools.save_candidate_voice(n,p,t)),
                inputs=[design_name,candidate_state,design_text],
                outputs=[live_role,design_status],
            )

        with gr.Tab("🧬 Voice Clone"):
            clone_source=gr.Radio(["Upload file","Remote upload"],value="Upload file",label="Source")
            clone_upload=gr.File(
                label="Upload audio / video",
                file_types=[".wav",".mp3",".m4a",".aac",".flac",".ogg",".opus",".mp4",".mkv",".webm",".mov",".avi",".m4v"],
            )
            with gr.Column(visible=False) as clone_remote_group:
                clone_url=gr.Textbox(label="Remote URL",placeholder="YouTube / TikTok / Facebook / X / Drive / MEGA / direct media")
                clone_cookie=gr.File(label="Cookies.txt (optional)")
                with gr.Accordion("Advanced headers",open=False):
                    clone_ua=gr.Textbox(label="User-Agent")
                    clone_referer=gr.Textbox(label="Referer")
            clone_source.change(
                lambda mode:(gr.update(visible=mode=="Upload file"),gr.update(visible=mode=="Remote upload")),
                inputs=clone_source,outputs=[clone_upload,clone_remote_group],
            )
            with gr.Row():
                clone_max=gr.Slider(5,28,20,1,label="Reference max seconds")
                clone_denoise=gr.Checkbox(value=False,label="Remove background noise")
            prep_clone=gr.Button("✨ Prepare clone reference")
            clone_ref_audio=gr.Audio(label="Prepared reference")
            clone_ref_state=gr.State("")
            clone_status=gr.Markdown()
            prep_clone.click(
                tools.prepare_clone_reference,
                inputs=[clone_source,clone_upload,clone_url,clone_cookie,clone_ua,clone_referer,clone_max,clone_denoise],
                outputs=[clone_ref_audio,clone_ref_state,clone_status],
            )
            with gr.Row():
                clone_language=gr.Dropdown(
                    ["Burmese / Myanmar","English","Thai","Chinese","Japanese"],
                    value="Burmese / Myanmar",label="Reference language"
                )
                transcribe_clone=gr.Button("📝 Transcribe reference")
            clone_transcript=gr.Textbox(lines=4,label="Reference transcript",placeholder="Automatic transcript appears here; correct names/slang before saving")
            transcribe_clone.click(
                tools.transcribe_clone_reference,
                inputs=[clone_ref_state,clone_language],
                outputs=[clone_transcript,clone_status],
            )
            clone_test_text=gr.Textbox(lines=3,label="Test cloned voice",value="မင်္ဂလာပါ။ ဒီအသံကို clone လုပ်ပြီး စမ်းသပ်နေပါတယ်။")
            test_clone=gr.Button("▶ Test cloned voice")
            clone_test_audio=gr.Audio(label="Clone test")
            test_clone.click(
                tools.test_clone_voice,
                inputs=[clone_ref_state,clone_transcript,clone_test_text,cfg,steps],
                outputs=[clone_test_audio,clone_status],
            )
            with gr.Row():
                clone_name=gr.Textbox(value="My Cloned Voice",label="Voice name")
                save_clone=gr.Button("💾 Save clone to library")
            save_clone.click(
                lambda n,w,t: _save_role_result(gr,tools.save_clone_voice(n,w,t)),
                inputs=[clone_name,clone_ref_state,clone_transcript],
                outputs=[live_role,clone_status],
            )

        with gr.Tab("💾 Library"):
            saved=gr.Dropdown(choices=tools.saved_voice_choices(),value=tools.saved_voice_choices()[0],label="Saved voice")
            with gr.Row():
                refresh=gr.Button("↻ Refresh")
                delete=gr.Button("🗑 Delete",variant="stop")
            lib_status=gr.Markdown()
            refresh.click(lambda:_refresh_result(gr),outputs=[saved])
            delete.click(lambda n:_delete_result(gr,n),inputs=saved,outputs=[saved,lib_status])

        # These are the values the built-in provider consumes during normal
        # pyVideoTrans dubbing. The main WebUI passes them into params before
        # constructing TransCreate.
        provider_cfg=cfg
        provider_steps=steps
        provider_delivery=delivery
        provider_custom=custom

    return {
        "panel":panel,
        "cfg":provider_cfg,
        "steps":provider_steps,
        "delivery":provider_delivery,
        "custom":provider_custom,
        "live_role":live_role,
        "profile":profile,
        "candidate_state":candidate_state,
        "design_text":design_text,
        "design_status":design_status,
        "use_candidate_btn":use_candidate_btn,
    }


def _save_role_result(gr,result):
    _choices,selected,status=result
    return gr.update(choices=tools.role_choices(),value=selected),status


def _refresh_result(gr):
    choices=tools.saved_voice_choices()
    return gr.update(choices=choices,value=choices[0])


def _delete_result(gr,name):
    choices,selected,status=tools.delete_saved_voice(name)
    return gr.update(choices=choices,value=selected),status


def build_video_editor(gr):
    """Post-translation editing room: subtitle blur/burn and logo overlay."""
    gr.Markdown("## 🎞 Video Editing Room\nUse the latest pyVideoTrans result or upload another video.")
    with gr.Row():
        source=gr.Video(label="Video",interactive=True)
        subtitle=gr.File(label="Generated subtitle (SRT/ASS/VTT)",file_types=[".srt",".ass",".vtt"])
    load_latest=gr.Button("↻ Load latest pyVideoTrans output")
    with gr.Row():
        blur=gr.Checkbox(False,label="Blur existing subtitles")
        blur_height=gr.Slider(8,45,22,1,label="Blur bottom area (%)")
        blur_strength=gr.Slider(1,30,10,1,label="Blur strength")
    burn=gr.Checkbox(True,label="Burn generated pyVideoTrans subtitles")
    with gr.Row():
        logo=gr.Image(type="filepath",label="Transparent logo / PNG")
        with gr.Column():
            logo_opacity=gr.Slider(0,1,.75,.05,label="Logo opacity")
            logo_width=gr.Slider(3,40,14,1,label="Logo width (%)")
            logo_pos=gr.Dropdown(["Top right","Top left","Bottom right","Bottom left"],value="Top right",label="Logo position")
    render=gr.Button("🎬 Render edited video",variant="primary")
    output=gr.Video(label="Edited video")
    output_file=gr.File(label="Download edited video")
    status=gr.Markdown()

    def _latest():
        v,s=latest_output()
        return v,s,"Loaded latest output." if v else "No output video found yet."

    def _render(v,s,b,bh,bs,burn_sub,logo_path,alpha,width,pos):
        sub_path=s if isinstance(s,str) else getattr(s,"name",None)
        out=render_edit_room(v,sub_path,blur_existing_subtitles=b,blur_height_percent=bh,blur_strength=bs,
                             burn_generated_subtitles=burn_sub,logo_path=logo_path,logo_opacity=alpha,
                             logo_width_percent=width,logo_position=pos)
        return out,out,f"Rendered: {Path(out).name}"

    load_latest.click(_latest,outputs=[source,subtitle,status])
    render.click(_render,inputs=[source,subtitle,blur,blur_height,blur_strength,burn,logo,logo_opacity,logo_width,logo_pos],
                 outputs=[output,output_file,status])


def build_remote_video_source(gr):
    """Video input with explicit import for remote URLs.

    Remote media is downloaded when the user clicks Import now, not when the
    translation pipeline starts. The translated job then reuses the imported
    local file.
    """
    source_mode = gr.Radio(
        ["Upload file", "Remote upload"],
        value="Upload file",
        label="Video source",
    )

    upload = gr.Video(
        label="Choose video file",
        interactive=True,
    )

    imported_path = gr.State("")
    imported_url = gr.State("")

    with gr.Column(visible=False) as remote_group:
        remote_url = gr.Textbox(
            label="Remote URL",
            placeholder="YouTube / TikTok / Facebook / X / Drive / MEGA / direct URL",
        )

        with gr.Row():
            import_now = gr.Button(
                "⬇ Import now",
                variant="primary",
            )
            import_status = gr.Markdown(
                "Paste a URL, then click **Import now**."
            )

        remote_preview = gr.Video(
            label="Imported remote video",
            interactive=False,
        )

        cookie = gr.File(
            label="Cookies.txt (optional)"
        )

        with gr.Accordion(
            "Advanced remote headers",
            open=False,
        ):
            ua = gr.Textbox(
                label="User-Agent"
            )
            referer = gr.Textbox(
                label="Referer"
            )

    def _switch_source(mode):
        return (
            gr.update(
                visible=mode == "Upload file"
            ),
            gr.update(
                visible=mode == "Remote upload"
            ),
        )

    source_mode.change(
        _switch_source,
        inputs=source_mode,
        outputs=[
            upload,
            remote_group,
        ],
    )

    def _import_remote(
        url,
        cookie_file,
        user_agent,
        referer_value,
    ):
        url = str(
            url or ""
        ).strip()

        if not url:
            raise gr.Error(
                "Paste a remote URL first."
            )

        try:
            local_path = download_remote_media(
                url,
                cookie_file=cookie_file,
                user_agent=str(
                    user_agent or ""
                ),
                referer=str(
                    referer_value or ""
                ),
            )

            local_path = str(
                Path(local_path).resolve()
            )

            return (
                local_path,
                local_path,
                url,
                (
                    "✅ Imported and ready. "
                    f"**{Path(local_path).name}** will be used when you click Start."
                ),
            )

        except Exception as exc:
            raise gr.Error(
                f"Remote import failed: {exc}"
            )

    import_now.click(
        _import_remote,
        inputs=[
            remote_url,
            cookie,
            ua,
            referer,
        ],
        outputs=[
            remote_preview,
            imported_path,
            imported_url,
            import_status,
        ],
    )

    return {
        "mode": source_mode,
        "upload": upload,
        "remote_group": remote_group,
        "url": remote_url,
        "cookie": cookie,
        "ua": ua,
        "referer": referer,
        "imported_path": imported_path,
        "imported_url": imported_url,
        "import_now": import_now,
        "import_status": import_status,
        "remote_preview": remote_preview,
    }

