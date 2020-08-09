//https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-c-extension
//python setup.py install
#include <torch/extension.h>
#include <vector>

torch::Tensor matmul_cuda(torch::Tensor input1, torch::Tensor input2, torch::Tensor output);
torch::Tensor vecmul_cuda(torch::Tensor input1, torch::Tensor input2, torch::Tensor output);

// C++ interface
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor matmul(torch::Tensor input1, torch::Tensor input2, torch::Tensor output)
{
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    return matmul_cuda(input1, input2, output);
}

torch::Tensor vecmul(torch::Tensor input1, torch::Tensor input2, torch::Tensor output)
{
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    return vecmul_cuda(input1, input2, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul, "Matrix Multiplication (CUDA)");
  m.def("vecmul", &vecmul, "Vector and Matrix Multiplication (CUDA)");
}
