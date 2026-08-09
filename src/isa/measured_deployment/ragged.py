"""Vectorized member-curve sampling from compact ragged state lists."""

from __future__ import annotations

import torch


def sample_member_ids(
    state_ids: torch.Tensor,
    member_ids_by_state: torch.Tensor,
    state_offsets: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Uniformly draw one member per state id without a Python state loop.

    All tensors must be on the same device, and ``generator`` must be created
    for that device.  The draw is intended to run once per Monte Carlo seed;
    the returned curve ids are then fixed for the full validation set.
    """
    if state_ids.dtype not in (torch.int16, torch.int32, torch.int64):
        raise TypeError("state_ids must use an integer dtype")
    if member_ids_by_state.ndim != 1 or state_offsets.ndim != 1:
        raise ValueError("member_ids_by_state and state_offsets must be 1-D")
    if state_offsets.numel() < 2:
        raise ValueError("state_offsets must contain at least two entries")
    device = state_ids.device
    if member_ids_by_state.device != device or state_offsets.device != device:
        raise ValueError("all sampler tensors must be on the same device")
    flat_state = state_ids.reshape(-1).long()
    if flat_state.numel() == 0:
        return torch.empty_like(state_ids, dtype=torch.int32)
    state_count = state_offsets.numel() - 1
    if int(flat_state.min()) < 0 or int(flat_state.max()) >= state_count:
        raise IndexError("state id is outside the ragged member table")
    starts = state_offsets[flat_state].long()
    counts = (state_offsets[flat_state + 1] - state_offsets[flat_state]).long()
    if bool(torch.any(counts <= 0)):
        raise ValueError("every referenced state must contain at least one member")
    random_value = torch.rand(
        flat_state.shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    rank = torch.floor(random_value * counts.float()).long()
    sampled = member_ids_by_state[starts + rank]
    return sampled.reshape(state_ids.shape).to(torch.int32)
