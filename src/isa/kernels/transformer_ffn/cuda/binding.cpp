#include <torch/extension.h>

#include <vector>

torch::Tensor ekv_forward_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor table_i,
    double delta_min,
    double delta_max,
    double v_sat);

torch::Tensor ekv_grad_x_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor grad_out,
    torch::Tensor table_i,
    torch::Tensor table_ddelta,
    double delta_min,
    double delta_max,
    double v_sat);

torch::Tensor ekv_grad_x_shared_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor grad_out,
    torch::Tensor table_i,
    torch::Tensor table_ddelta,
    double delta_min,
    double delta_max,
    double v_sat);

std::vector<torch::Tensor> ekv_grad_w_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor grad_out,
    torch::Tensor table_ddelta,
    double delta_min,
    double delta_max,
    double v_sat);

std::vector<torch::Tensor> ekv_grad_w_shared_cuda(
    torch::Tensor x,
    torch::Tensor wpos,
    torch::Tensor wneg,
    torch::Tensor grad_out,
    torch::Tensor table_ddelta,
    double delta_min,
    double delta_max,
    double v_sat);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &ekv_forward_cuda, "EKV LUT forward (CUDA)");
    module.def("grad_x", &ekv_grad_x_cuda, "EKV LUT grad_x (CUDA)");
    module.def("grad_x_shared", &ekv_grad_x_shared_cuda, "EKV LUT grad_x, shared LUT (CUDA)");
    module.def("grad_w", &ekv_grad_w_cuda, "EKV LUT grad_w (CUDA)");
    module.def("grad_w_shared", &ekv_grad_w_shared_cuda, "EKV LUT grad_w, shared LUT (CUDA)");
}
