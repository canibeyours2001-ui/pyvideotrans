def test_scheduler_demo_log_contains_worker_ready_and_progress(capsys):
    from videotrans.tts.voxcpm2_scheduler import run_demo

    run_demo()
    out = capsys.readouterr().out

    assert "VoxCPM2 worker 0 ready on cuda:0" in out
    assert "VoxCPM2 worker 1 ready on cuda:1" in out
    assert "Completed: 5 / 5" in out
