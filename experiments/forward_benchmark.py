"""Small, data-loading-free forward benchmark for converted models."""

from __future__ import annotations

import statistics
import time

import torch


@torch.inference_mode()
def benchmark_forward(call, device: str, repeats: int = 3, warmup: int = 2) -> dict:
    """Measure one already prepared input, synchronizing CUDA at each boundary.

    The result describes inference on this host and input shape. It does not
    include tokenization or image decoding, and is not neuromorphic latency.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    def sync():
        if cuda:
            torch.cuda.synchronize()

    for _ in range(warmup):
        out = call()
        sync()
        del out

    sync()
    base_bytes = torch.cuda.memory_allocated() if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times_ms = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        out = call()
        sync()
        times_ms.append(1000.0 * (time.perf_counter() - start))
        del out
    peak_bytes = torch.cuda.max_memory_allocated() if cuda else None
    return {
        "median_forward_ms": statistics.median(times_ms),
        "forward_ms": times_ms,
        "peak_gpu_allocated_bytes": peak_bytes,
        "peak_gpu_extra_bytes": (max(peak_bytes - base_bytes, 0)
                                 if cuda else None),
        "baseline_gpu_allocated_bytes": base_bytes,
        "warmup": warmup,
        "repeats": repeats,
    }
