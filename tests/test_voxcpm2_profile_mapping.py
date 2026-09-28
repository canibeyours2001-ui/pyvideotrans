def test_profile_lookup_accepts_display_name_and_id():
    from videotrans.tts.voxcpm2_profiles import get_profile

    by_id = get_profile("movie_recap_fast")
    by_name = get_profile("🎬 Movie Recap — Fast Pace")

    assert by_id == by_name
    assert by_id.profile_id == "movie_recap_fast"
