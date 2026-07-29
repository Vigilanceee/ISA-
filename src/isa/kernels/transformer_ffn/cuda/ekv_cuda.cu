#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

struct LutPair {
    float base;
    float derivative;
};

__device__ __forceinline__ float clamp_lut_coordinate(float value, int size) {
    return fminf(fmaxf(value, 0.0f), static_cast<float>(size - 1));
}

__device__ __forceinline__ float lut_lookup(
    const float* __restrict__ table,
    float delta,
    float delta_min,
    float scale,
    int size) {
    const float coordinate = clamp_lut_coordinate((delta - delta_min) * scale, size);
    const int i0 = __float2int_rz(coordinate);
    const int i1 = min(i0 + 1, size - 1);
    const float alpha = coordinate - static_cast<float>(i0);
    const float q0 = __ldg(table + i0);
    const float q1 = __ldg(table + i1);
    return fmaf(alpha, q1 - q0, q0);
}

__device__ __forceinline__ LutPair lut_pair(
    const float* __restrict__ table_i,
    const float* __restrict__ table_ddelta,
    float delta,
    float delta_min,
    float scale,
    int size) {
    const float coordinate = clamp_lut_coordinate((delta - delta_min) * scale, size);
    const int i0 = __float2int_rz(coordinate);
    const int i1 = min(i0 + 1, size - 1);
    const float alpha = coordinate - static_cast<float>(i0);
    const float base0 = __ldg(table_i + i0);
    const float base1 = __ldg(table_i + i1);
    const float grad0 = __ldg(table_ddelta + i0);
    const float grad1 = __ldg(table_ddelta + i1);
    return {
        fmaf(alpha, base1 - base0, base0),
        fmaf(alpha, grad1 - grad0, grad0),
    };
}

__device__ __forceinline__ float lut_lookup_shared(
    const float* table,
    float delta,
    float delta_min,
    float scale,
    int size) {
    const float coordinate = clamp_lut_coordinate((delta - delta_min) * scale, size);
    const int i0 = __float2int_rz(coordinate);
    const int i1 = min(i0 + 1, size - 1);
    const float alpha = coordinate - static_cast<float>(i0);
    return fmaf(alpha, table[i1] - table[i0], table[i0]);
}

__device__ __forceinline__ LutPair lut_pair_shared(
    const float* table_i,
    const float* table_ddelta,
    float delta,
    float delta_min,
    float scale,
    int size) {
    const float coordinate = clamp_lut_coordinate((delta - delta_min) * scale, size);
    const int i0 = __float2int_rz(coordinate);
    const int i1 = min(i0 + 1, size - 1);
    const float alpha = coordinate - static_cast<float>(i0);
    return {
        fmaf(alpha, table_i[i1] - table_i[i0], table_i[i0]),
        fmaf(alpha, table_ddelta[i1] - table_ddelta[i0], table_ddelta[i0]),
    };
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void ekv_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wpos,
    const float* __restrict__ wneg,
    const float* __restrict__ table_i,
    float* __restrict__ output,
    int m_size,
    int k_size,
    int o_size,
    int table_size,
    float delta_min,
    float delta_scale,
    float v_sat) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;
    const int output_index = blockIdx.x * warps_per_block + warp_in_block;
    const int output_size = m_size * o_size;
    if (output_index >= output_size) {
        return;
    }

    const int m = output_index / o_size;
    const int o = output_index - m * o_size;
    const float* x_row = x + static_cast<int64_t>(m) * k_size;
    const float* wp_row = wpos + static_cast<int64_t>(o) * k_size;
    const float* wn_row = wneg + static_cast<int64_t>(o) * k_size;
    float accumulator = 0.0f;
    for (int k = lane; k < k_size; k += kWarpSize) {
        const float xv = x_row[k];
        const float positive = lut_lookup(
            table_i, xv - wp_row[k], delta_min, delta_scale, table_size);
        const float negative = lut_lookup(
            table_i, xv - wn_row[k], delta_min, delta_scale, table_size);
        accumulator += (positive - negative) / (1.0f + xv / v_sat);
    }
    accumulator = warp_sum(accumulator);
    if (lane == 0) {
        output[output_index] = accumulator;
    }
}

__global__ void ekv_grad_x_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wpos,
    const float* __restrict__ wneg,
    const float* __restrict__ grad_out,
    const float* __restrict__ table_i,
    const float* __restrict__ table_ddelta,
    float* __restrict__ grad_x,
    int m_size,
    int k_size,
    int o_size,
    int table_size,
    float delta_min,
    float delta_scale,
    float v_sat) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t output_size = static_cast<int64_t>(m_size) * k_size;
    if (index >= output_size) {
        return;
    }

    const int m = index / k_size;
    const int k = index - static_cast<int64_t>(m) * k_size;
    const float xv = x[index];
    const float denominator = 1.0f + xv / v_sat;
    const float inv_denominator = 1.0f / denominator;
    const float denominator_term = 1.0f / (v_sat * denominator * denominator);
    const float* grad_row = grad_out + static_cast<int64_t>(m) * o_size;
    float accumulator = 0.0f;

    for (int o = 0; o < o_size; ++o) {
        const int64_t weight_index = static_cast<int64_t>(o) * k_size + k;
        const LutPair positive = lut_pair(
            table_i,
            table_ddelta,
            xv - wpos[weight_index],
            delta_min,
            delta_scale,
            table_size);
        const LutPair negative = lut_pair(
            table_i,
            table_ddelta,
            xv - wneg[weight_index],
            delta_min,
            delta_scale,
            table_size);
        const float derivative =
            (positive.derivative - negative.derivative) * inv_denominator -
            (positive.base - negative.base) * denominator_term;
        accumulator = fmaf(grad_row[o], derivative, accumulator);
    }
    grad_x[index] = accumulator;
}

__global__ void ekv_grad_x_shared_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wpos,
    const float* __restrict__ wneg,
    const float* __restrict__ grad_out,
    const float* __restrict__ table_i,
    const float* __restrict__ table_ddelta,
    float* __restrict__ grad_x,
    int m_size,
    int k_size,
    int o_size,
    int table_size,
    float delta_min,
    float delta_scale,
    float v_sat) {
    extern __shared__ float shared_lut[];
    float* shared_i = shared_lut;
    float* shared_ddelta = shared_lut + table_size;
    for (int i = threadIdx.x; i < table_size; i += blockDim.x) {
        shared_i[i] = table_i[i];
        shared_ddelta[i] = table_ddelta[i];
    }
    __syncthreads();

    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t output_size = static_cast<int64_t>(m_size) * k_size;
    if (index >= output_size) {
        return;
    }
    const int m = index / k_size;
    const int k = index - static_cast<int64_t>(m) * k_size;
    const float xv = x[index];
    const float denominator = 1.0f + xv / v_sat;
    const float inv_denominator = 1.0f / denominator;
    const float denominator_term = 1.0f / (v_sat * denominator * denominator);
    const float* grad_row = grad_out + static_cast<int64_t>(m) * o_size;
    float accumulator = 0.0f;
    for (int o = 0; o < o_size; ++o) {
        const int64_t weight_index = static_cast<int64_t>(o) * k_size + k;
        const LutPair positive = lut_pair_shared(
            shared_i,
            shared_ddelta,
            xv - wpos[weight_index],
            delta_min,
            delta_scale,
            table_size);
        const LutPair negative = lut_pair_shared(
            shared_i,
            shared_ddelta,
            xv - wneg[weight_index],
            delta_min,
            delta_scale,
            table_size);
        const float derivative =
            (positive.derivative - negative.derivative) * inv_denominator -
            (positive.base - negative.base) * denominator_term;
        accumulator = fmaf(grad_row[o], derivative, accumulator);
    }
    grad_x[index] = accumulator;
}

__global__ void ekv_grad_w_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wpos,
    const float* __restrict__ wneg,
    const float* __restrict__ grad_out,
    const float* __restrict__ table_ddelta,
    float* __restrict__ grad_wpos,
    float* __restrict__ grad_wneg,
    int m_size,
    int k_size,
    int o_size,
    int table_size,
    float delta_min,
    float delta_scale,
    float v_sat) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t output_size = static_cast<int64_t>(o_size) * k_size;
    if (index >= output_size) {
        return;
    }

    const int o = index / k_size;
    const int k = index - static_cast<int64_t>(o) * k_size;
    const float wp = wpos[index];
    const float wn = wneg[index];
    float accumulator_pos = 0.0f;
    float accumulator_neg = 0.0f;
    for (int m = 0; m < m_size; ++m) {
        const float xv = x[static_cast<int64_t>(m) * k_size + k];
        const float grad = grad_out[static_cast<int64_t>(m) * o_size + o];
        const float inv_denominator = 1.0f / (1.0f + xv / v_sat);
        const float grad_pos = lut_lookup(
            table_ddelta, xv - wp, delta_min, delta_scale, table_size);
        const float grad_neg = lut_lookup(
            table_ddelta, xv - wn, delta_min, delta_scale, table_size);
        accumulator_pos = fmaf(grad, -grad_pos * inv_denominator, accumulator_pos);
        accumulator_neg = fmaf(grad, grad_neg * inv_denominator, accumulator_neg);
    }
    grad_wpos[index] = accumulator_pos;
    grad_wneg[index] = accumulator_neg;
}

__global__ void ekv_grad_w_shared_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wpos,
    const float* __restrict__ wneg,
    const float* __restrict__ grad_out,
    const float* __restrict__ table_ddelta,
    float* __restrict__ grad_wpos,
    float* __restrict__ grad_wneg,
    int m_size,
    int k_size,
    int o_size,
    int table_size,
    float delta_min,
    float delta_scale,
    float v_sat) {
    extern __shared__ float shared_ddelta[];
    for (int i = threadIdx.x; i < table_size; i += blockDim.x) {
        shared_ddelta[i] = table_ddelta[i];
    }
    __syncthreads();

    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t output_size = static_cast<int64_t>(o_size) * k_size;
    if (index >= output_size) {
        return;
    }
    const int o = index / k_size;
    const int k = index - static_cast<int64_t>(o) * k_size;
    const float wp = wpos[index];
    const float wn = wneg[index];
    float accumulator_pos = 0.0f;
    float accumulator_neg = 0.0f;
    for (int m = 0; m < m_size; ++m) {
        const float xv = x[static_cast<int64_t>(m) * k_size + k];
        const float grad = grad_out[static_cast<int64_t>(m) * o_size + o];
        const float inv_denominator = 1.0f / (1.0f + xv / v_sat);
        const float grad_pos = lut_lookup_shared(
            shared_ddelta, xv - wp, delta_min, delta_scale, table_size);
        const float grad_neg = lut_lookup_shared(
            shared_ddelta, xv - wn, delta_min, delta_scale, table_size);
        accumulator_pos = fmaf(grad, -grad_pos * inv_denominator, accumulator_pos);
        accumulator_neg = fmaf(grad, grad_neg * inv_denominator, accumulator_neg);
    }
    grad_wpos[index] = accumulator_pos;
    grad_wneg[index] = accumulator_neg;
}

void check_float_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor ekv_forward_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor table_i,
    double delta_min,
    double delta_max,
    double v_sat) {
    check_float_cuda_contiguous(x, "x");
    check_float_cuda_contiguous(wpos, "wpos");
    check_float_cuda_contiguous(wneg, "wneg");
    check_float_cuda_contiguous(table_i, "table_i");
    c10::cuda::CUDAGuard device_guard(x.device());
    const int m = x.size(0);
    const int k = x.size(1);
    const int o = wpos.size(0);
    const int table_size = table_i.numel();
    auto output = torch::empty({m, o}, x.options().dtype(torch::kFloat32));
    const int warps_per_block = kThreads / kWarpSize;
    const int blocks = (m * o + warps_per_block - 1) / warps_per_block;
    const float scale = static_cast<float>((table_size - 1) / (delta_max - delta_min));
    ekv_forward_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        wpos.data_ptr<float>(),
        wneg.data_ptr<float>(),
        table_i.data_ptr<float>(),
        output.data_ptr<float>(),
        m,
        k,
        o,
        table_size,
        static_cast<float>(delta_min),
        scale,
        static_cast<float>(v_sat));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor ekv_grad_x_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor grad_out,
    torch::Tensor table_i,
    torch::Tensor table_ddelta,
    double delta_min,
    double delta_max,
    double v_sat) {
    check_float_cuda_contiguous(x, "x");
    check_float_cuda_contiguous(wpos, "wpos");
    check_float_cuda_contiguous(wneg, "wneg");
    check_float_cuda_contiguous(grad_out, "grad_out");
    check_float_cuda_contiguous(table_i, "table_i");
    check_float_cuda_contiguous(table_ddelta, "table_ddelta");
    c10::cuda::CUDAGuard device_guard(x.device());
    const int m = x.size(0);
    const int k = x.size(1);
    const int o = wpos.size(0);
    const int table_size = table_i.numel();
    auto grad_x = torch::empty_like(x, x.options().dtype(torch::kFloat32));
    const int64_t elements = static_cast<int64_t>(m) * k;
    const int blocks = (elements + kThreads - 1) / kThreads;
    const float scale = static_cast<float>((table_size - 1) / (delta_max - delta_min));
    ekv_grad_x_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        wpos.data_ptr<float>(),
        wneg.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        table_i.data_ptr<float>(),
        table_ddelta.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        m,
        k,
        o,
        table_size,
        static_cast<float>(delta_min),
        scale,
        static_cast<float>(v_sat));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}

torch::Tensor ekv_grad_x_shared_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor grad_out,
    torch::Tensor table_i,
    torch::Tensor table_ddelta,
    double delta_min,
    double delta_max,
    double v_sat) {
    check_float_cuda_contiguous(x, "x");
    check_float_cuda_contiguous(wpos, "wpos");
    check_float_cuda_contiguous(wneg, "wneg");
    check_float_cuda_contiguous(grad_out, "grad_out");
    check_float_cuda_contiguous(table_i, "table_i");
    check_float_cuda_contiguous(table_ddelta, "table_ddelta");
    c10::cuda::CUDAGuard device_guard(x.device());
    const int m = x.size(0);
    const int k = x.size(1);
    const int o = wpos.size(0);
    const int table_size = table_i.numel();
    TORCH_CHECK(table_size <= 8192, "shared grad_x supports LUT_DELTA_SIZE <= 8192");
    auto grad_x = torch::empty_like(x, x.options().dtype(torch::kFloat32));
    const int64_t elements = static_cast<int64_t>(m) * k;
    const int blocks = (elements + kThreads - 1) / kThreads;
    const float scale = static_cast<float>((table_size - 1) / (delta_max - delta_min));
    const size_t shared_bytes = static_cast<size_t>(table_size) * 2 * sizeof(float);
    if (shared_bytes > 48 * 1024) {
        const cudaError_t status = cudaFuncSetAttribute(
            ekv_grad_x_shared_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        TORCH_CHECK(status == cudaSuccess, "could not opt in to 64 KiB shared memory: ", cudaGetErrorString(status));
    }
    ekv_grad_x_shared_kernel<<<
        blocks, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        wpos.data_ptr<float>(),
        wneg.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        table_i.data_ptr<float>(),
        table_ddelta.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        m,
        k,
        o,
        table_size,
        static_cast<float>(delta_min),
        scale,
        static_cast<float>(v_sat));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
}

std::vector<torch::Tensor> ekv_grad_w_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor grad_out,
    torch::Tensor table_ddelta,
    double delta_min,
    double delta_max,
    double v_sat) {
    check_float_cuda_contiguous(x, "x");
    check_float_cuda_contiguous(wpos, "wpos");
    check_float_cuda_contiguous(wneg, "wneg");
    check_float_cuda_contiguous(grad_out, "grad_out");
    check_float_cuda_contiguous(table_ddelta, "table_ddelta");
    c10::cuda::CUDAGuard device_guard(x.device());
    const int m = x.size(0);
    const int k = x.size(1);
    const int o = wpos.size(0);
    const int table_size = table_ddelta.numel();
    auto grad_wpos = torch::empty_like(wpos, wpos.options().dtype(torch::kFloat32));
    auto grad_wneg = torch::empty_like(wneg, wneg.options().dtype(torch::kFloat32));
    const int64_t elements = static_cast<int64_t>(o) * k;
    const int blocks = (elements + kThreads - 1) / kThreads;
    const float scale = static_cast<float>((table_size - 1) / (delta_max - delta_min));
    ekv_grad_w_kernel<<<blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        wpos.data_ptr<float>(),
        wneg.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        table_ddelta.data_ptr<float>(),
        grad_wpos.data_ptr<float>(),
        grad_wneg.data_ptr<float>(),
        m,
        k,
        o,
        table_size,
        static_cast<float>(delta_min),
        scale,
        static_cast<float>(v_sat));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_wpos, grad_wneg};
}

std::vector<torch::Tensor> ekv_grad_w_shared_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor grad_out,
    torch::Tensor table_ddelta,
    double delta_min,
    double delta_max,
    double v_sat) {
    check_float_cuda_contiguous(x, "x");
    check_float_cuda_contiguous(wpos, "wpos");
    check_float_cuda_contiguous(wneg, "wneg");
    check_float_cuda_contiguous(grad_out, "grad_out");
    check_float_cuda_contiguous(table_ddelta, "table_ddelta");
    c10::cuda::CUDAGuard device_guard(x.device());
    const int m = x.size(0);
    const int k = x.size(1);
    const int o = wpos.size(0);
    const int table_size = table_ddelta.numel();
    TORCH_CHECK(table_size <= 8192, "shared grad_w supports LUT_DELTA_SIZE <= 8192");
    auto grad_wpos = torch::empty_like(wpos, wpos.options().dtype(torch::kFloat32));
    auto grad_wneg = torch::empty_like(wneg, wneg.options().dtype(torch::kFloat32));
    const int64_t elements = static_cast<int64_t>(o) * k;
    const int blocks = (elements + kThreads - 1) / kThreads;
    const float scale = static_cast<float>((table_size - 1) / (delta_max - delta_min));
    const size_t shared_bytes = static_cast<size_t>(table_size) * sizeof(float);
    ekv_grad_w_shared_kernel<<<
        blocks, kThreads, shared_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        wpos.data_ptr<float>(),
        wneg.data_ptr<float>(),
        grad_out.data_ptr<float>(),
        table_ddelta.data_ptr<float>(),
        grad_wpos.data_ptr<float>(),
        grad_wneg.data_ptr<float>(),
        m,
        k,
        o,
        table_size,
        static_cast<float>(delta_min),
        scale,
        static_cast<float>(v_sat));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_wpos, grad_wneg};
}
