import time
from dataclasses import dataclass

import pytest


@dataclass
class FakeWorker:
    worker_id: int
    device: str
    durations: dict[int, float]
    events: list

    def generate(self, job):
        start = time.monotonic()
        self.events.append(("start", self.device, job.chunk_id, start))
        time.sleep(self.durations[job.chunk_id])
        end = time.monotonic()
        self.events.append(("finish", self.device, job.chunk_id, end))
        return f"audio-{job.chunk_id}"


def test_two_workers_use_dynamic_queue_and_restore_original_order():
    from videotrans.tts.voxcpm2_scheduler import VoxCPM2ChunkJob, VoxCPM2Scheduler

    events = []
    durations = {1: 0.18, 2: 0.03, 3: 0.03, 4: 0.08, 5: 0.03}
    workers = [FakeWorker(0, "cuda:0", durations, events), FakeWorker(1, "cuda:1", durations, events)]
    jobs = [VoxCPM2ChunkJob(chunk_id=i, text=f"chunk {i}", output_path=f"{i}.wav") for i in range(1, 6)]

    results = VoxCPM2Scheduler(workers).run(jobs)

    assert [r.chunk_id for r in results] == [1, 2, 3, 4, 5]
    assert [r.audio for r in results] == ["audio-1", "audio-2", "audio-3", "audio-4", "audio-5"]

    starts = [e for e in events if e[0] == "start"]
    assert starts[0][2] == 1
    assert starts[1][2] == 2
    assert abs(starts[0][3] - starts[1][3]) < 0.05

    finish_2 = next(e for e in events if e[0] == "finish" and e[2] == 2)
    start_3 = next(e for e in events if e[0] == "start" and e[2] == 3)
    finish_1 = next(e for e in events if e[0] == "finish" and e[2] == 1)
    assert start_3[1] == finish_2[1]
    assert start_3[3] < finish_1[3]


def test_failed_chunk_reports_index_without_deadlock():
    from videotrans.tts.voxcpm2_scheduler import VoxCPM2ChunkJob, VoxCPM2Scheduler, VoxCPM2SchedulerError

    class FailingWorker(FakeWorker):
        def generate(self, job):
            if job.chunk_id == 2:
                raise RuntimeError("boom")
            return super().generate(job)

    events = []
    workers = [FailingWorker(0, "cuda:0", {1: 0.01, 3: 0.01}, events)]
    jobs = [VoxCPM2ChunkJob(chunk_id=i, text=f"chunk {i}", output_path=f"{i}.wav") for i in range(1, 4)]

    with pytest.raises(VoxCPM2SchedulerError) as exc:
        VoxCPM2Scheduler(workers).run(jobs)

    assert exc.value.chunk_id == 2
    assert "boom" in str(exc.value)


def test_scheduler_stops_claiming_jobs_after_cancellation():
    from videotrans.tts.voxcpm2_scheduler import VoxCPM2ChunkJob, VoxCPM2Scheduler

    worker = FakeWorker(0, "cpu", {1: 0, 2: 0, 3: 0}, [])
    calls = 0

    def should_stop():
        nonlocal calls
        calls += 1
        return calls > 1

    results = VoxCPM2Scheduler([worker]).run(
        [VoxCPM2ChunkJob(i, f"chunk {i}", f"{i}.wav") for i in range(1, 4)],
        should_stop=should_stop,
    )

    assert [result.chunk_id for result in results] == [1]
