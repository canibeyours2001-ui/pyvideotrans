def test_long_chunk_is_split_on_punctuation_boundaries():
    from videotrans.tts._voxcpm2 import split_long_text

    text = "Sentence one. Sentence two! Sentence three? Sentence four."

    parts = split_long_text(text, max_chars=25)

    assert parts == ["Sentence one.", "Sentence two!", "Sentence three?", "Sentence four."]
