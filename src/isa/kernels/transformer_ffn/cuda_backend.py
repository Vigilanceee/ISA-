"""JIT loader for the shared Flash-EKV CUDA backward extension."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_extension(verbose: bool = False):
    from torch.utils.cpp_extension import load

    build_dir = Path(
        os.environ.get(
            "EKV_CUDA_BUILD_DIR",
            str(ROOT / ".torch_extensions" / "isa_ekv_cuda_shared_v2"),
        )
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="isa_ekv_cuda_shared_v2",
        sources=[
            str(ROOT / "cuda" / "binding.cpp"),
            str(ROOT / "cuda" / "ekv_cuda.cu"),
        ],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
        ],
        with_cuda=True,
        verbose=verbose,
    )


def prebuild() -> None:
    extension = load_extension(verbose=os.environ.get("EKV_CUDA_VERBOSE_BUILD", "0") == "1")
    print(f"[ekv-cuda] ready: {extension.__name__}")


def shared_backward(
    x: torch.Tensor,
    wpos: torch.Tensor,
    wneg: torch.Tensor,
    grad_out: torch.Tensor,
    table_i: torch.Tensor,
    table_ddelta: torch.Tensor,
    delta_min: float,
    delta_max: float,
    v_sat: float,
    need_grad_x: bool,
    need_grad_w: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    extension = load_extension()
    x32 = x.float().contiguous()
    wpos32 = wpos.float().contiguous()
    wneg32 = wneg.float().contiguous()
    grad32 = grad_out.float().contiguous()
    table_i = table_i.float().contiguous()
    table_ddelta = table_ddelta.float().contiguous()

    grad_x = None
    grad_wpos = None
    grad_wneg = None
    if need_grad_x:
        grad_x = extension.grad_x_shared(
            x32,
            wpos32,
            wneg32,
            grad32,
            table_i,
            table_ddelta,
            delta_min,
            delta_max,
            v_sat,
        ).to(dtype=x.dtype)
    if need_grad_w:
        grad_wpos, grad_wneg = extension.grad_w_shared(
            x32,
            wpos32,
            wneg32,
            grad32,
            table_ddelta,
            delta_min,
            delta_max,
            v_sat,
        )
        grad_wpos = grad_wpos.to(dtype=wpos.dtype)
        grad_wneg = grad_wneg.to(dtype=wneg.dtype)
    return grad_x, grad_wpos, grad_wneg
