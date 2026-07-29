"""Device-aware EKV initialization utilities."""

from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from isa.operators.cim import CIMLinear, PhysicalFFN, HybridFFN, VoltageMapping


def _iter_cim_layers(model: nn.Module) -> List[Tuple[str, CIMLinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, CIMLinear)]


def _iter_voltage_mappings(model: nn.Module) -> List[Tuple[str, VoltageMapping]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, VoltageMapping)]


def _iter_physical_ffns(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return all PhysicalFFN and HybridFFN modules (both have clamp_vth)."""
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, (PhysicalFFN, HybridFFN))]


def load_shared_baseline_weights(model: nn.Module, baseline_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Load only shape-compatible baseline tensors into the physical/hybrid model."""
    physical_state = model.state_dict()
    compatible = {
        name: value
        for name, value in baseline_state.items()
        if name in physical_state and physical_state[name].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return {
        "baseline_shared_tensors": len(compatible),
        "baseline_shared_parameters": int(sum(value.numel() for value in compatible.values())),
    }


def _collect_baseline_ffn_pairs(
    baseline_model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
    max_rows: int,
    use_amp: bool,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    inputs: Dict[str, List[torch.Tensor]] = {}
    outputs: Dict[str, List[torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, args, output):
            x = args[0].detach().reshape(-1, args[0].shape[-1])
            y = output.detach().reshape(-1, output.shape[-1])
            remaining = max_rows - sum(t.shape[0] for t in inputs[name])
            if remaining > 0:
                inputs[name].append(x[:remaining].float().cpu())
                outputs[name].append(y[:remaining].float().cpu())
        return hook

    for name, module in baseline_model.named_modules():
        if not name.endswith(".mlp"):
            continue
        inputs[name] = []
        outputs[name] = []
        handles.append(module.register_forward_hook(make_hook(name)))

    baseline_model.eval()
    with torch.no_grad():
        for i, (images, _targets) in enumerate(loader):
            if i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                baseline_model(images)

    for handle in handles:
        handle.remove()
    return {
        name: (torch.cat(inputs[name], dim=0), torch.cat(outputs[name], dim=0))
        for name in inputs
        if inputs[name]
    }


def match_physical_ffns_to_baseline(
    model: nn.Module,
    baseline_model: nn.Module,
    loader,
    device: torch.device,
    steps: int = 20,
    max_batches: int = 2,
    max_rows: int = 4096,
    batch_size: int = 256,
    lr: float = 1e-3,
    use_amp: bool = False,
    seed: int = 42,
) -> Dict[str, float | str]:
    """Calibrate each CIM/Hybrid FFN against a baseline FFN before classification training."""
    if steps <= 0:
        return {"baseline_match_steps": 0}
    pairs = _collect_baseline_ffn_pairs(
        baseline_model, loader, device, max_batches=max_batches, max_rows=max_rows, use_amp=use_amp
    )
    physical_modules = dict(model.named_modules())
    initial_losses = []
    final_losses = []
    layer_records = []
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for name, (inputs, targets) in pairs.items():
        module = physical_modules.get(name)
        if not isinstance(module, (PhysicalFFN, HybridFFN)):
            continue
        module.train()
        optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.0)
        rows = inputs.shape[0]
        first_loss = None
        last_loss = None
        for _ in range(steps):
            indices = torch.randint(rows, (min(batch_size, rows),), generator=generator)
            x = inputs[indices].to(device, non_blocking=True).unsqueeze(0)
            target = targets[indices].to(device, non_blocking=True).unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = module(x)
                loss = F.mse_loss(output.float(), target.float())
            loss.backward()
            optimizer.step()
            module.clamp_vth()
            value = float(loss.detach().item())
            first_loss = value if first_loss is None else first_loss
            last_loss = value
        initial_losses.append(float(first_loss))
        final_losses.append(float(last_loss))
        layer_records.append(f"{name}:{first_loss:.6g}->{last_loss:.6g}")

    model.train()
    if not final_losses:
        return {"baseline_match_steps": steps, "baseline_match_layers": 0}
    return {
        "baseline_match_steps": steps,
        "baseline_match_layers": len(final_losses),
        "baseline_match_initial_mse": float(sum(initial_losses) / len(initial_losses)),
        "baseline_match_final_mse": float(sum(final_losses) / len(final_losses)),
        "baseline_match_by_layer": ";".join(layer_records),
    }


def _collect_voltage_mapping_inputs(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
    max_rows: int,
    use_amp: bool,
) -> Dict[str, torch.Tensor]:
    mappings = _iter_voltage_mappings(model)
    inputs: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in mappings}
    handles = []

    def make_hook(name: str):
        def hook(_module, args):
            x = args[0].detach()
            flat = x.reshape(-1, x.shape[-1])
            remaining = max_rows - sum(t.shape[0] for t in inputs[name])
            if remaining > 0:
                inputs[name].append(flat[:remaining].float().cpu())
        return hook

    for name, module in mappings:
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (images, _targets) in enumerate(loader):
            if i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                model(images)

    for h in handles:
        h.remove()

    return {name: torch.cat(vals, dim=0) for name, vals in inputs.items() if vals}


@torch.no_grad()
def _calibrate_voltage_mappings(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
    max_rows: int,
    use_amp: bool,
    quantile: float,
    v_low: float,
    v_high: float,
    iters: int,
) -> Dict[str, float | str]:
    q_hi = min(max(float(quantile), 0.5), 0.9999)
    q_lo = max(0.0, 1.0 - q_hi)
    records: Dict[str, Tuple[float, float, float, float]] = {}

    for _ in range(max(1, int(iters))):
        raw_inputs = _collect_voltage_mapping_inputs(model, loader, device, max_batches, max_rows, use_amp)
        for name, module in _iter_voltage_mappings(model):
            if name not in raw_inputs:
                continue
            x = raw_inputs[name].float().flatten()
            lo = torch.quantile(x, q_lo).item()
            hi = torch.quantile(x, q_hi).item()
            span = max(hi - lo, 1e-6)
            out_low = max(0.0, min(float(v_low), float(module.voltage_max)))
            out_high = max(out_low + 1e-6, min(float(v_high), float(module.voltage_max)))
            scale = (out_high - out_low) / span
            shift = out_low - lo * scale
            module.set_affine(scale, shift)
            records[name] = (scale, shift, lo, hi)

    if not records:
        return {"voltage_map_init": "data", "voltage_map_layers": 0}
    scales = [v[0] for v in records.values()]
    shifts = [v[1] for v in records.values()]
    return {
        "voltage_map_init": "data",
        "voltage_map_layers": len(records),
        "voltage_map_scale_mean": float(sum(scales) / len(scales)),
        "voltage_map_shift_mean": float(sum(shifts) / len(shifts)),
        "voltage_map_by_layer": ";".join(
            f"{k}:scale={v[0]:.6g},shift={v[1]:.6g},lo={v[2]:.6g},hi={v[3]:.6g}"
            for k, v in records.items()
        ),
    }


def _uniform_js(probs: torch.Tensor) -> float:
    mean_p = probs.mean(dim=0).clamp_min(1e-8)
    mean_p = mean_p / mean_p.sum()
    uni = torch.full_like(mean_p, 1.0 / mean_p.numel())
    mid = 0.5 * (mean_p + uni)
    js = 0.5 * (mean_p * (mean_p / mid).log()).sum() + 0.5 * (uni * (uni / mid).log()).sum()
    return float(js.item())


def _collect_layer_inputs(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
    max_rows: int,
    use_amp: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    layers = _iter_cim_layers(model)
    inputs: Dict[str, List[torch.Tensor]] = {name: [] for name, _ in layers}
    handles = []

    def make_hook(name: str):
        def hook(_module, args, _output):
            x = args[0].detach()
            flat = x.reshape(-1, x.shape[-1])
            remaining = max_rows - sum(t.shape[0] for t in inputs[name])
            if remaining > 0:
                inputs[name].append(flat[:remaining].float().cpu())
        return hook

    for name, module in layers:
        handles.append(module.register_forward_hook(make_hook(name)))

    logits_list: List[torch.Tensor] = []
    target_list: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
            logits_list.append(logits.detach().float().cpu())
            target_list.append(targets.detach().cpu())

    for h in handles:
        h.remove()

    cat_inputs = {name: torch.cat(vals, dim=0) for name, vals in inputs.items() if vals}
    return cat_inputs, torch.cat(logits_list, dim=0), torch.cat(target_list, dim=0)


def _set_global_init(model: nn.Module, strategy: str, eps_k: float) -> None:
    symmetric = strategy not in {"random_fixed", "random_reverse_tia"}
    mode = "random" if not symmetric else "centered_symmetric"
    for _name, layer in _iter_cim_layers(model):
        layer.reset_vth(center=None, eps_k=eps_k, symmetric=symmetric, mode=mode)
        layer.set_r_tia(float(layer.cfg.get("R_TIA", 1e5)))


def _center_from_inputs(model: nn.Module, layer_inputs: Dict[str, torch.Tensor], eps_k: float) -> None:
    for name, layer in _iter_cim_layers(model):
        if name not in layer_inputs:
            continue
        center = layer_inputs[name].mean().item()
        layer.reset_vth(center=center, eps_k=eps_k, symmetric=True, mode="centered_symmetric")


def _reverse_tia(
    model: nn.Module,
    layer_inputs: Dict[str, torch.Tensor],
    rho: float,
    quantile: float,
    max_rows: int,
) -> Dict[str, float]:
    rtias = {}
    for name, layer in _iter_cim_layers(model):
        if name not in layer_inputs:
            continue
        x = layer_inputs[name].to(next(layer.parameters()).device)
        idiff = layer.estimate_idiff(x, max_rows=max_rows).detach().abs().flatten()
        q = torch.quantile(idiff.float(), quantile).clamp_min(1e-12)
        v_min = float(layer.cfg.get("V_signed_min", -4.0))
        v_max = float(layer.cfg.get("V_signed_max", 4.0))
        target_amp = 0.5 * (v_max - v_min) * float(rho)
        r_tia = float(target_amp / q.item())
        layer.set_r_tia(r_tia)
        rtias[name] = r_tia
    return rtias


def _clip_ratios(model: nn.Module, layer_inputs: Dict[str, torch.Tensor], max_rows: int) -> Dict[str, float]:
    ratios = {}
    for name, layer in _iter_cim_layers(model):
        if name not in layer_inputs:
            continue
        x = layer_inputs[name].to(next(layer.parameters()).device)
        idiff = layer.estimate_idiff(x, max_rows=max_rows).detach().float()
        vout = idiff * layer.r_tia.to(device=idiff.device, dtype=idiff.dtype)
        v_min = float(layer.cfg.get("V_signed_min", -4.0))
        v_max = float(layer.cfg.get("V_signed_max", 4.0))
        ratios[name] = float(((vout <= v_min) | (vout >= v_max)).float().mean().item())
    return ratios


def _evaluate_init(model: nn.Module, loader, device: torch.device, max_batches: int, max_rows: int, use_amp: bool) -> Dict[str, float | str]:
    layer_inputs, logits, targets = _collect_layer_inputs(model, loader, device, max_batches, max_rows, use_amp)
    loss = F.cross_entropy(logits, targets).item()
    probs = logits.softmax(dim=-1)
    js = _uniform_js(probs)
    clips = _clip_ratios(model, layer_inputs, max_rows=max_rows)
    max_clip = max(clips.values()) if clips else 0.0
    mean_clip = sum(clips.values()) / max(len(clips), 1)
    return {
        "init_loss": float(loss),
        "init_loss_delta": float(abs(loss - math.log(logits.shape[-1]))),
        "init_js": float(js),
        "init_clip_max": float(max_clip),
        "init_clip_mean": float(mean_clip),
        "init_clip_by_layer": ";".join(f"{k}:{v:.6f}" for k, v in clips.items()),
    }


def initialize_ekv_model(
    model: nn.Module,
    loader,
    device: torch.device,
    strategy: str,
    eps_k: float = 0.5,
    candidates: int = 1,
    rho: float = 0.8,
    quantile: float = 0.99,
    clip_threshold: float = 0.05,
    max_batches: int = 1,
    max_rows: int = 4096,
    seed: int = 42,
    use_amp: bool = False,
    voltage_map_init: str = "data",
    voltage_map_quantile: float = 0.995,
    voltage_map_low: float = 0.2,
    voltage_map_high: float = 3.8,
    voltage_map_iters: int = 2,
) -> Dict[str, float | str]:
    if strategy == "default" or not _iter_cim_layers(model):
        return {"ekv_init": strategy}

    if strategy not in {
        "random_fixed",
        "centered_fixed",
        "random_reverse_tia",
        "centered_reverse_tia",
        "calibrated",
    }:
        raise ValueError(f"Unknown EKV init strategy: {strategy}")

    mapping_metrics: Dict[str, float | str] = {}
    if voltage_map_init == "data":
        mapping_metrics = _calibrate_voltage_mappings(
            model, loader, device, max_batches=max_batches, max_rows=max_rows,
            use_amp=use_amp, quantile=voltage_map_quantile,
            v_low=voltage_map_low, v_high=voltage_map_high,
            iters=voltage_map_iters,
        )
    elif voltage_map_init not in {"identity", "none", ""}:
        raise ValueError(f"Unknown voltage_map_init: {voltage_map_init}")

    num_candidates = max(1, int(candidates if strategy == "calibrated" else 1))
    best_score = None
    best_state = None
    best_metrics: Dict[str, float | str] = {}

    for cand in range(num_candidates):
        torch.manual_seed(seed + cand)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + cand)

        base_strategy = "centered_reverse_tia" if strategy == "calibrated" else strategy
        _set_global_init(model, base_strategy, eps_k)

        needs_center = base_strategy in {"centered_fixed", "centered_reverse_tia"}
        needs_reverse = base_strategy in {"random_reverse_tia", "centered_reverse_tia"}

        if needs_center or needs_reverse:
            layer_inputs, _logits, _targets = _collect_layer_inputs(model, loader, device, max_batches, max_rows, use_amp)
            if needs_center:
                _center_from_inputs(model, layer_inputs, eps_k)
                layer_inputs, _logits, _targets = _collect_layer_inputs(model, loader, device, max_batches, max_rows, use_amp)
            rtias = _reverse_tia(model, layer_inputs, rho, quantile, max_rows) if needs_reverse else {}
        else:
            rtias = {}

        metrics = _evaluate_init(model, loader, device, max_batches, max_rows, use_amp)
        metrics.update(mapping_metrics)
        metrics.update({
            "ekv_init": strategy,
            "ekv_candidate": cand,
            "ekv_eps_k": float(eps_k),
            "ekv_rho": float(rho),
            "ekv_quantile": float(quantile),
            "ekv_r_tia_by_layer": ";".join(f"{k}:{v:.6g}" for k, v in rtias.items()),
        })
        feasible = float(metrics["init_clip_max"]) <= clip_threshold
        score = (0 if feasible else 1, float(metrics["init_js"]), float(metrics["init_loss_delta"]))
        if best_score is None or score < best_score:
            best_score = score
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics
