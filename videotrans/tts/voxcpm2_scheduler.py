from __future__ import annotations

import argparse
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from videotrans.tts.voxcpm2_voice import SelectedVoice


@dataclass(frozen=True, slots=True)
class VoxCPM2ChunkJob:
    chunk_id: int
    text: str
    output_path: str | Path
    selected_voice: SelectedVoice | None = None


@dataclass(frozen=True, slots=True)
class VoxCPM2ChunkResult:
    chunk_id: int
    audio: str
    output_path: Path


class VoxCPM2WorkerProtocol(Protocol):
    worker_id: int
    device: str

    def generate(self, job: VoxCPM2ChunkJob) -> str: ...


class VoxCPM2SchedulerError(RuntimeError):
    def __init__(self, chunk_id: int, message: str):
        self.chunk_id = chunk_id
        super().__init__(f"VoxCPM2 chunk {chunk_id} failed: {message}")


class VoxCPM2Scheduler:
    def __init__(self, workers: Sequence[VoxCPM2WorkerProtocol]):
        if not workers:
            raise ValueError("At least one VoxCPM2 worker is required")
        self.workers = list(workers)

    def run(self, jobs: list[VoxCPM2ChunkJob]) -> list[VoxCPM2ChunkResult]:
        work_queue: queue.Queue[VoxCPM2ChunkJob] = queue.Queue()
        result_queue: queue.Queue[VoxCPM2ChunkResult | VoxCPM2SchedulerError] = queue.Queue()
        for job in jobs:
            work_queue.put(job)

        def consume(worker: VoxCPM2WorkerProtocol) -> None:
            while True:
                try:
                    job = work_queue.get_nowait()
                except queue.Empty:
                    return
                print(f"[{worker.device}] chunk {job.chunk_id} started", flush=True)
                try:
                    audio = worker.generate(job)
                    print(f"[{worker.device}] chunk {job.chunk_id} finished", flush=True)
                    result_queue.put(VoxCPM2ChunkResult(job.chunk_id, audio, Path(job.output_path)))
                except Exception as exc:  # worker-thread boundary: report instead of hanging scheduler
                    result_queue.put(VoxCPM2SchedulerError(job.chunk_id, str(exc)))
                finally:
                    work_queue.task_done()

        threads = [threading.Thread(target=consume, args=(worker,), daemon=True) for worker in self.workers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        results: list[VoxCPM2ChunkResult] = []
        while not result_queue.empty():
            item = result_queue.get()
            if isinstance(item, VoxCPM2SchedulerError):
                raise item
            results.append(item)
        if len(results) != len(jobs):
            raise VoxCPM2SchedulerError(-1, f"scheduler finished with {len(results)} / {len(jobs)} chunk results")
        return sorted(results, key=lambda result: result.chunk_id)


@dataclass(slots=True)
class _DemoWorker:
    worker_id: int
    device: str
    durations: dict[int, float]

    def generate(self, job: VoxCPM2ChunkJob) -> str:
        time.sleep(self.durations[job.chunk_id])
        return f"audio-{job.chunk_id}"


def run_demo() -> None:
    durations = {1: 0.18, 2: 0.03, 3: 0.03, 4: 0.08, 5: 0.03}
    workers = [_DemoWorker(0, "cuda:0", durations), _DemoWorker(1, "cuda:1", durations)]
    for worker in workers:
        print(f"VoxCPM2 worker {worker.worker_id} ready on {worker.device}")
    jobs = [VoxCPM2ChunkJob(i, f"chunk {i}", f"{i}.wav") for i in range(1, 6)]
    results = VoxCPM2Scheduler(workers).run(jobs)
    print(f"Completed: {len(results)} / {len(jobs)}")
    print("ordered: " + ",".join(str(result.chunk_id) for result in results))


def main() -> int:
    parser = argparse.ArgumentParser(description="VoxCPM2 dynamic scheduler")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
