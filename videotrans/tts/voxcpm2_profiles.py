from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class VoxCPM2Profile:
    profile_id: str
    display_name: str
    instruction: str


VOXCPM2_PROFILES: Final[tuple[VoxCPM2Profile, ...]] = (
    VoxCPM2Profile("u_aung", "👨 U Aung — Deep Calm Male", "An adult Burmese man in his late thirties to mid forties. A naturally low baritone voice with warm chest resonance, full but relaxed tone, calm confidence, and mature presence. Speak in authentic conversational Burmese with clean articulation, natural sentence melody, realistic micro-pauses, and subtle breathing. Use a relaxed medium-slow pace. Keep the performance warm and human, never robotic, overly dramatic, theatrical, or like a commercial announcer. Maintain consistent vocal weight while allowing small natural variations in pitch, rhythm, emphasis, and emotion."),
    VoxCPM2Profile("ko_min", "👨 Ko Min — Friendly Young Male", "A Burmese man in his mid to late twenties with a naturally friendly, approachable medium-pitched voice. The tone should be warm, relaxed, youthful, and conversational without sounding childish. Use authentic everyday Burmese rhythm, smooth phrasing, clear consonants, natural vowel length, and gentle pitch variation. Speak at a comfortable medium pace with small realistic pauses. The delivery should feel like a real person speaking directly to a friend, not reading from a script and not performing like an announcer."),
    VoxCPM2Profile("u_htet", "👴 U Htet — Wise Older Male", "A Burmese man around sixty years old with a mature, naturally deep voice. The voice has gentle age texture and warmth but remains healthy and clear. Use a wise, patient, reassuring storyteller delivery with measured pacing. Allow thoughtful pauses between important ideas and subtle emotional shading. Pronunciation should remain natural and distinctly Burmese. Avoid exaggerated elderly characteristics, excessive raspiness, stage acting, or artificial dramatic emphasis."),
    VoxCPM2Profile("ko_zaw", "👨 Ko Zaw — Energetic Presenter", "A Burmese man around thirty years old with a clear, energetic, confident speaking voice. The voice should sound naturally enthusiastic rather than loud or exaggerated. Use lively but controlled pitch movement, crisp articulation, smooth connected speech, and strong sentence momentum. Speak slightly faster than normal conversation while keeping every word easy to understand. Use short natural pauses and varied emphasis to maintain attention. Avoid shouting, radio-announcer exaggeration, or synthetic excitement."),
    VoxCPM2Profile("daw_may", "👩 Daw May — Warm Adult Female", "A Burmese woman in her late thirties to mid forties with a warm, gentle, naturally feminine voice. The tone should feel caring, grounded, trustworthy, and emotionally intelligent. Use soft but clear articulation, comfortable chest resonance, smooth sentence transitions, and realistic breathing. Speak at a relaxed medium pace with natural Burmese intonation. Avoid an overly sweet, cartoonish, whispery, commercial, or theatrical performance."),
    VoxCPM2Profile("ma_su", "👩 Ma Su — Bright Young Female", "A Burmese woman in her early to mid twenties with a youthful, bright and naturally pleasant voice. She sounds cheerful, friendly, expressive and approachable without becoming childish or overly cute. Use light natural energy, clear pronunciation, smooth connected speech, and realistic conversational pitch movement. Speak at a medium pace with spontaneous-sounding micro-pauses. Avoid anime-like exaggeration, high-pitched squealing, artificial sweetness, or robotic rhythm."),
    VoxCPM2Profile("daw_thida", "👩 Daw Thida — Elegant Mature Female", "A mature Burmese woman with a smooth, composed and elegant voice. The vocal tone should be refined, confident, warm and professionally controlled without becoming formal or distant. Use excellent articulation, balanced resonance, natural Burmese phrasing, subtle pitch variation and smooth sentence endings. Speak at a measured medium pace. The performance should sound sophisticated and human, not like advertising copy or a synthetic newsreader."),
    VoxCPM2Profile("ma_nandar", "👩 Ma Nandar — Soft Emotional Female", "A Burmese woman in her late twenties to early thirties with a soft, intimate and emotionally expressive voice. The delivery should feel close, personal and sincere while remaining clearly audible and naturally voiced. Use gentle breath control, subtle changes in pitch and intensity, soft sentence endings and realistic pauses. Speak slightly slower than casual conversation. Avoid excessive whispering, melodrama, crying effects, or exaggerated emotional acting."),
    VoxCPM2Profile("news_presenter", "📰 Professional Burmese News Presenter", "A professional adult Burmese news presenter with a neutral, trustworthy and highly intelligible broadcast voice. Use balanced vocal resonance, precise pronunciation, stable volume and controlled sentence emphasis. Maintain a steady medium pace while allowing brief natural pauses between ideas. The voice should sound polished but still recognizably human, with subtle prosody instead of flat robotic reading. Avoid sensationalism, excessive authority, advertising tone, or dramatic movie-trailer delivery."),
    VoxCPM2Profile("documentary_narrator", "🎥 Deep Documentary Narrator", "A mature Burmese documentary narrator with a naturally deep, resonant voice. The performance should feel intelligent, observant, serious and quietly cinematic. Use controlled low-register resonance without artificially forcing the pitch downward. Speak at a deliberate medium-slow pace with excellent clarity, thoughtful pauses, subtle suspense and restrained emotional emphasis. Keep the narration natural and intimate rather than sounding like an exaggerated movie trailer."),
    VoxCPM2Profile("movie_recap_fast", "🎬 Movie Recap — Fast Pace", "A young adult Burmese male movie-recap narrator with a natural, energetic and highly engaging voice designed for fast-paced online storytelling. Speak noticeably faster than normal conversation while preserving clean pronunciation and complete intelligibility. Maintain strong forward momentum with short pauses, quick sentence transitions and frequent but natural changes in emphasis. Use controlled excitement, curiosity and tension according to the story. The narration should feel spontaneous and human, as if enthusiastically explaining a movie to a friend. Keep breaths efficient and unobtrusive. Avoid shouting, rushed slurring, artificial hype, repetitive pitch patterns, radio-announcer delivery or robotic speed-reading. Prioritize high listener retention while preserving believable Burmese prosody."),
)

PROFILE_BY_ID: Final[dict[str, VoxCPM2Profile]] = {profile.profile_id: profile for profile in VOXCPM2_PROFILES}
PROFILE_BY_NAME: Final[dict[str, VoxCPM2Profile]] = {profile.display_name: profile for profile in VOXCPM2_PROFILES}


def get_profile(value: str) -> VoxCPM2Profile:
    profile = PROFILE_BY_ID.get(value) or PROFILE_BY_NAME.get(value)
    if profile is None:
        raise KeyError(f"Unknown VoxCPM2 profile: {value}")
    return profile


def profile_names() -> list[str]:
    return [profile.display_name for profile in VOXCPM2_PROFILES]


def main() -> int:
    parser = argparse.ArgumentParser(description="VoxCPM2 voice profiles")
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()
    if args.list_profiles:
        for profile in VOXCPM2_PROFILES:
            print(f"{profile.profile_id}\t{profile.display_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
