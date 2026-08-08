import pytest


torch = pytest.importorskip("torch")


def _cuda_only():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def _reference_conv(x, w_pos, w_neg, params, fallback, kernel_size, stride, padding):
    import torch.nn.functional as F

    batch, _channels, height, width = x.shape
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    out_h = (height + 2 * ph - kh) // sh + 1
    out_w = (width + 2 * pw - kw) // sw + 1
    columns = F.unfold(x, kernel_size=kernel_size, stride=stride, padding=padding)
    flat = columns.permute(0, 2, 1).contiguous().view(batch * out_h * out_w, -1)
    output = fallback.apply(flat, w_pos, w_neg, params)
    return output.view(batch, out_h * out_w, w_pos.shape[0]).permute(0, 2, 1).reshape(
        batch, w_pos.shape[0], out_h, out_w
    )


@pytest.mark.parametrize("device_type", ("pcm", "fefet", "flash"))
def test_exact_direct_conv_matches_unfold_formula_forward_and_backward(device_type):
    _cuda_only()
    from isa.kernels.device_sweep.direct_conv_triton import DirectFormulaConv2d
    from isa.kernels.device_sweep.fefet_triton import FeFETFunction
    from isa.kernels.device_sweep.flash_triton import FlashFunction
    from isa.kernels.device_sweep.pcm_triton import PCMFunction

    params_by_device = {
        "pcm": {
            "I0_pcm": 3976.6,
            "I0_pcm_decay": 53.6321,
            "V_T_pcm": 0.02585,
        },
        "fefet": {
            "I_S": 6.3026,
            "n": 1.0280,
            "U_T": 0.026,
            "V_D": 0.1,
            "A_lk": 0.8214,
            "B_lk": 1.3171,
            "V_sat": 5.0,
            "raw_kernel_backend": "split",
            "raw_forward_backend": "split_k",
        },
        "flash": {
            "I_S": 0.75875,
            "n": 4.1360,
            "U_T": 0.026,
            "V_D": 0.1,
            "V_sat": 8.2624,
        },
    }
    fallback_by_device = {
        "pcm": PCMFunction,
        "fefet": FeFETFunction,
        "flash": FlashFunction,
    }
    params = params_by_device[device_type]
    fallback = fallback_by_device[device_type]
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    torch.manual_seed(20260808)
    x_base = torch.rand(2, 3, 7, 7, device="cuda", dtype=torch.float32)
    if device_type == "pcm":
        x_base = x_base - 0.5
        lo, hi = 0.15, 0.31
    elif device_type == "fefet":
        x_base = x_base * 4.0
        lo, hi = 1.8711, 3.7399
    else:
        x_base = x_base * 4.0
        lo, hi = 0.0, 5.0
    shape = (5, 3 * 3 * 3)
    wp_base = torch.empty(shape, device="cuda").uniform_(lo, hi)
    wn_base = torch.empty(shape, device="cuda").uniform_(lo, hi)

    x_direct = x_base.detach().clone().requires_grad_(True)
    wp_direct = wp_base.detach().clone().requires_grad_(True)
    wn_direct = wn_base.detach().clone().requires_grad_(True)
    direct = DirectFormulaConv2d.apply(
        x_direct,
        wp_direct,
        wn_direct,
        params,
        device_type,
        kernel_size,
        stride,
        padding,
        fallback,
    )

    x_reference = x_base.detach().clone().requires_grad_(True)
    wp_reference = wp_base.detach().clone().requires_grad_(True)
    wn_reference = wn_base.detach().clone().requires_grad_(True)
    reference = _reference_conv(
        x_reference,
        wp_reference,
        wn_reference,
        params,
        fallback,
        kernel_size,
        stride,
        padding,
    )
    assert torch.allclose(direct, reference, rtol=3.0e-4, atol=2.0e-4)

    gradient = torch.randn_like(reference)
    direct.backward(gradient)
    reference.backward(gradient)
    assert torch.allclose(x_direct.grad, x_reference.grad, rtol=5.0e-4, atol=3.0e-4)
    assert torch.allclose(wp_direct.grad, wp_reference.grad, rtol=5.0e-4, atol=3.0e-4)
    assert torch.allclose(wn_direct.grad, wn_reference.grad, rtol=5.0e-4, atol=3.0e-4)


def test_reused_ffn_tile_matches_flash_device_formula():
    _cuda_only()
    from isa.kernels.device_sweep.flash_triton import FlashFunction
    from isa.kernels.transformer_ffn.ekv_triton import TritonEKVMatmulFn

    params = {
        "I_S": 0.75875,
        "n": 4.1360,
        "U_T": 0.026,
        "V_D": 0.1,
        "V_sat": 8.2624,
    }
    torch.manual_seed(20260808)
    x_base = torch.rand(17, 27, device="cuda") * 4.0
    wp_base = torch.rand(13, 27, device="cuda") * 5.0
    wn_base = torch.rand(13, 27, device="cuda") * 5.0
    tensors = []
    outputs = []
    for function in (FlashFunction, TritonEKVMatmulFn):
        x = x_base.detach().clone().requires_grad_(True)
        wp = wp_base.detach().clone().requires_grad_(True)
        wn = wn_base.detach().clone().requires_grad_(True)
        output = function.apply(x, wp, wn, params)
        output.square().mean().backward()
        tensors.append((x, wp, wn))
        outputs.append(output)
    assert torch.allclose(outputs[0], outputs[1], rtol=5.0e-5, atol=2.0e-4)
    for legacy, tiled in zip(tensors[0], tensors[1]):
        assert torch.allclose(legacy.grad, tiled.grad, rtol=5.0e-5, atol=2.0e-4)


def test_flash_split_reduction_matches_flash_device_formula():
    _cuda_only()
    from isa.kernels.device_sweep.flash_triton import FlashFunction, FlashSplitFunction

    params = {
        "I_S": 0.75875,
        "n": 4.1360,
        "U_T": 0.026,
        "V_D": 0.1,
        "V_sat": 8.2624,
    }
    torch.manual_seed(20260808)
    x_base = torch.rand(31, 67, device="cuda") * 4.0
    wp_base = torch.rand(19, 67, device="cuda") * 5.0
    wn_base = torch.rand(19, 67, device="cuda") * 5.0
    gradient = torch.randn(31, 19, device="cuda")
    tensors = []
    outputs = []
    for function in (FlashFunction, FlashSplitFunction):
        x = x_base.detach().clone().requires_grad_(True)
        wp = wp_base.detach().clone().requires_grad_(True)
        wn = wn_base.detach().clone().requires_grad_(True)
        output = function.apply(x, wp, wn, params)
        output.backward(gradient)
        tensors.append((x, wp, wn))
        outputs.append(output)
    assert torch.allclose(outputs[0], outputs[1], rtol=5.0e-5, atol=3.0e-4)
    for legacy, split in zip(tensors[0], tensors[1]):
        assert torch.allclose(legacy.grad, split.grad, rtol=5.0e-5, atol=3.0e-4)


def test_pcm_split_reduction_matches_legacy_exact_formula():
    _cuda_only()
    from isa.kernels.device_sweep.pcm_triton import PCMFunction

    base = {
        "I0_pcm": 3976.6,
        "I0_pcm_decay": 53.6321,
        "V_T_pcm": 0.02585,
    }
    torch.manual_seed(20260808)
    x_base = torch.rand(31, 67, device="cuda") - 0.5
    wp_base = torch.empty(19, 67, device="cuda").uniform_(0.15, 0.31)
    wn_base = torch.empty(19, 67, device="cuda").uniform_(0.15, 0.31)
    gradient = torch.randn(31, 19, device="cuda")
    tensors = []
    outputs = []
    for raw_forward, raw_backward in (("legacy", "legacy"), ("split_k", "split")):
        params = dict(
            base,
            raw_forward_backend=raw_forward,
            raw_kernel_backend=raw_backward,
        )
        x = x_base.detach().clone().requires_grad_(True)
        wp = wp_base.detach().clone().requires_grad_(True)
        wn = wn_base.detach().clone().requires_grad_(True)
        output = PCMFunction.apply(x, wp, wn, params)
        output.backward(gradient)
        tensors.append((x, wp, wn))
        outputs.append(output)
    assert torch.allclose(outputs[0], outputs[1], rtol=5.0e-5, atol=3.0e-4)
    for legacy, split in zip(tensors[0], tensors[1]):
        assert torch.allclose(legacy.grad, split.grad, rtol=5.0e-5, atol=3.0e-4)
