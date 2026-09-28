def test_webui_exports_import_and_voxcpm_helpers():
    import webui

    assert webui.CLI_LANG == "en"
    assert "Local File" in webui.MEDIA_SOURCE_CHOICES
    assert "Direct Download URL" in webui.MEDIA_SOURCE_CHOICES
    assert "YouTube" in webui.MEDIA_SOURCE_CHOICES
    assert callable(webui.import_media_for_webui)
    assert callable(webui.test_voxcpm2_voice_for_webui)
    assert callable(webui.use_voxcpm2_voice_for_webui)


def test_english_option_labels_preserve_chinese_language_support():
    import webui

    assert "No subtitles" in webui.SUBTITLE_TYPES
    assert "Hard subtitles" in webui.SUBTITLE_TYPES
    assert any("中文" in str(name) or "Chinese" in str(name) for name in webui.LANG_DISPLAY_NAMES)
