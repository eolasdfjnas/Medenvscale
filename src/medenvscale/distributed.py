from __future__ import annotations

import os
from typing import Any


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1") or 1)


def global_rank() -> int:
    return int(os.environ.get("RANK", "0") or 0)


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0") or 0)


def is_distributed() -> bool:
    return world_size() > 1


def is_main_process() -> bool:
    return global_rank() == 0


def setup_torch_distributed_device() -> None:
    if not is_distributed():
        return
    try:
        import torch  # type: ignore
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())


def barrier() -> None:
    if not is_distributed():
        return
    try:
        import torch.distributed as dist  # type: ignore
    except ImportError:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def distributed_metadata() -> dict[str, Any]:
    return {
        "distributed": is_distributed(),
        "world_size": world_size(),
        "rank": global_rank(),
        "local_rank": local_rank(),
    }
