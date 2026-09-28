def test_safe_filename_strips_path_traversal():
    from videotrans.remote_importer import safe_filename

    assert safe_filename("../../secret.mp4") == "secret.mp4"
    assert safe_filename("bad/name?.mp4") == "name.mp4"
    assert safe_filename("") == "download.bin"
