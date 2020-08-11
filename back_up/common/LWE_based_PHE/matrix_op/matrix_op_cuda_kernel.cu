#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

    template <typename scalar_t>
    __global__ void matmul_cuda_kernel(
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input1,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input2,
            int k,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output
            )
    {
        for (int i = 0; i < k; ++i)
        {
            output[blockIdx.x * 16 + threadIdx.x][blockIdx.y * 16 + threadIdx.y]
                += input1[blockIdx.x * 16 + threadIdx.x][i] *
                    input2[i][blockIdx.y * 16 + threadIdx.y];
        }
    }

    template <typename scalar_t>
    __global__ void vecmul_cuda_kernel(
            torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> input1,
            torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input2,
            int m,
            torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> output
            )
    {
        for (int i = 0; i < m; ++i)
        {
            output[blockIdx.x * 16 + threadIdx.x]
                += input1[i] * input2[i][blockIdx.x * 16 + threadIdx.x];
        }
    }

}

torch::Tensor matmul_cuda(torch::Tensor input1, torch::Tensor input2, torch::Tensor output)
{
    const auto size_m = input1.size(0);
    const auto size_n = input2.size(1);
    int size_k = input1.size(1);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(size_m / 16, size_n / 16);

    AT_DISPATCH_ALL_TYPES(input1.type(), "matmul_cuda", ([&] {
        matmul_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        input1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
        input2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
        size_k,
        output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
  }));
  return output;
}

torch::Tensor vecmul_cuda(torch::Tensor input1, torch::Tensor input2, torch::Tensor output)
{
    int size_m = input1.size(0);
    const auto size_n = input2.size(1);

    dim3 threadsPerBlock(16, 1);
    dim3 numBlocks(size_n / 16, 1);

    AT_DISPATCH_ALL_TYPES(input1.type(), "vecmul_cuda", ([&] {
        vecmul_cuda_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
        input1.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
        input2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
        size_m,
        output.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>());
  }));
  return output;
}