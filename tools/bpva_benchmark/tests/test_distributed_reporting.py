import json
import time
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from tools.bpva_benchmark.reporting import create_run_session, write_partial


class FakeAccelerator:
    def __init__(self, rank: int):
        self.process_index = rank
        self.num_processes = 2
        self.is_main_process = rank == 0


def _reporting_worker(rank: int, init_path: str, output_dir: str) -> None:
    distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
    )
    try:
        session = create_run_session(
            output_dir, FakeAccelerator(rank), exact=True
        )
        write_partial(
            session,
            kind="data",
            phase="measure",
            completed=1,
            total=2,
            records=[],
            metadata={"rank": rank},
        )
        distributed.barrier()
    finally:
        distributed.destroy_process_group()


def test_two_rank_cpu_gloo_session_broadcast_and_partials(tmp_path):
    init_file = tmp_path / "dist-init"
    output_dir = tmp_path / "run"
    context = multiprocessing.spawn(
        _reporting_worker,
        args=(str(init_file), str(output_dir)),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 30
    while not context.join(timeout=1):
        if time.monotonic() >= deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
            raise AssertionError("双 rank Gloo reporting 测试超时")

    partials = [
        json.loads(
            (output_dir / "partial" / f"rank-{rank:05d}.json").read_text()
        )
        for rank in range(2)
    ]
    manifest = json.loads((output_dir / "manifest.json").read_text())

    assert manifest["status"] == "running"
    assert {item["rank"] for item in partials} == {0, 1}
    assert {item["generation"] for item in partials} == {
        manifest["generation"]
    }
    assert {item["metadata"]["rank"] for item in partials} == {0, 1}
    assert {manifest["output_dir"]} == {str(output_dir.resolve())}
