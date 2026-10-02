from __future__ import annotations

from pathlib import Path

from videotrans.voxcpm2 import gradio_tools as tools
from videotrans.webui_media import latest_output, render_edit_room


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
            live_btn=gr.Button("⚡ Generate + Live Stream",variant="primary")
            live_audio=gr.Audio(label="Live combined preview",streaming=True,autoplay=True)
            live_status=gr.Markdown()
            live_file=gr.File(label="Final WAV")
            def _stream(script,role,delivery_name,custom_text,speed_value,cfg_value,step_value,chars,pause):
                instruction=delivery_instruction(delivery_name)
                yield from tools.stream_voiceover(script,role,instruction,custom_text,speed_value,cfg_value,step_value,chars,pause)
            live_btn.click(
                _stream,
                inputs=[live_script,live_role,delivery,custom,speed,cfg,steps,chunk_chars,join_pause],
                outputs=[live_audio,live_status,live_file],
            )

        with gr.Tab("🎭 Voice Design"):
            profile=gr.Dropdown(choices=tools.profile_choices(),value=tools.profile_choices()[0],label="Voice profile")
            design_text=gr.Textbox(lines=4,label="Audition text",value="မင်္ဂလာပါ။ ဒီအသံကို စမ်းသပ်နေပါတယ်။")
            design_btn=gr.Button("🎲 Generate profile candidate")
            design_audio=gr.Audio(label="Candidate")
            candidate_state=gr.State("")
            design_status=gr.Markdown()
            with gr.Row():
                design_name=gr.Textbox(value="My Burmese Voice",label="Save name")
                save_design=gr.Button("💾 Save this voice")
            save_design.click(
                lambda n,p,t: _save_result(gr,tools.save_candidate_voice(n,p,t)),
                inputs=[design_name,candidate_state,design_text],
                outputs=[live_role,live_role,design_status],
            )
            design_btn.click(
                tools.generate_profile_candidate,
                inputs=[profile,design_text,cfg,steps],
                outputs=[design_audio,candidate_state,design_status],
            )

        with gr.Tab("🧬 Voice Clone"):
            clone_source=gr.Radio(["Upload file","Remote upload"],value="Upload file",label="Source")
            clone_upload=gr.File(label="Upload audio / video",file_types=["audio","video"])
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
            clone_transcript=gr.Textbox(lines=4,label="Reference transcript",placeholder="Paste/correct exactly what the speaker says")
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
                lambda n,w,t: _save_result(gr,tools.save_clone_voice(n,w,t)),
                inputs=[clone_name,clone_ref_state,clone_transcript],
                outputs=[live_role,live_role,clone_status],
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
    }


def _save_result(gr,result):
    choices,selected,status=result
    return gr.update(choices=choices,value=selected),gr.update(choices=tools.role_choices(),value=selected),status


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
    source_mode=gr.Radio(["Upload file","Remote upload"],value="Upload file",label="Video source")
    upload=gr.Video(label="Choose video file",interactive=True)
    with gr.Column(visible=False) as remote_group:
        remote_url=gr.Textbox(label="Remote URL",placeholder="YouTube / TikTok / Facebook / X / Drive / MEGA / direct URL")
        cookie=gr.File(label="Cookies.txt (optional)")
        with gr.Accordion("Advanced remote headers",open=False):
            ua=gr.Textbox(label="User-Agent")
            referer=gr.Textbox(label="Referer")
    source_mode.change(
        lambda mode:(gr.update(visible=mode=="Upload file"),gr.update(visible=mode=="Remote upload")),
        inputs=source_mode,outputs=[upload,remote_group],
    )
    return {"mode":source_mode,"upload":upload,"remote_group":remote_group,"url":remote_url,"cookie":cookie,"ua":ua,"referer":referer}
