#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace gs_kernels {

__global__ void project2d_kernel(int numel, const float* pt3d, const float* m4x4, float eps, float* result) {
    // each thread process a row in pt3d
    // res(n, 0) = pt3d[n] dot m4x4[:, 0] / (pt3d[n] dot m4x4[:, -1] + eps)
    // res(n, 1) = pt3d[n] dot m4x4[:, 1] / (pt3d[n] dot m4x4[:, -1] + eps)

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= numel) {
        return;
    }

    // create shared memory and allocate the whole m4x4 matrix
    __shared__ float m[16];

    if (tid < 16) {
        m[tid] = m4x4[tid];
    }

    __syncthreads();

    int offset = gid * 3;
    // calculate z
    float z = pt3d[offset] * m[3] + pt3d[offset + 1] * m[7] + pt3d[offset + 2] * m[11] + m[15] + eps;

    // calculate x, y
    float x = pt3d[offset] * m[0] + pt3d[offset + 1] * m[4] + pt3d[offset + 2] * m[8] + m[12];
    float y = pt3d[offset] * m[1] + pt3d[offset + 1] * m[5] + pt3d[offset + 2] * m[9] + m[13];

    result[gid* 2] = x / z;
    result[gid* 2 + 1] = y / z;
}

at::Tensor project2d_cuda(const at::Tensor& points_3d, const at::Tensor& proj_matrix, double eps) {
    int N = points_3d.sizes()[0];
    TORCH_CHECK(points_3d.sizes() == at::IntArrayRef({N, 3}));
    TORCH_CHECK(points_3d.dtype() == at::kFloat);
    TORCH_CHECK(proj_matrix.sizes() == at::IntArrayRef({4, 4}));
    TORCH_CHECK(proj_matrix.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(points_3d.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(proj_matrix.device().type() == at::DeviceType::CUDA);

    at::Tensor pt_contig = points_3d.contiguous();
    at::Tensor m_contig = proj_matrix.contiguous();
    at::Tensor result = at::empty(at::IntArrayRef({N, 2}), pt_contig.options());

    const float* pt_ptr = pt_contig.data_ptr<float>();
    const float* m_ptr = m_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    project2d_kernel<<<(N + 255) / 256, 256, 0, stream>>>(N, pt_ptr, m_ptr, eps, result_ptr);

    return result;
}

TORCH_LIBRARY_IMPL(gs_kernels, CUDA, m) {
    m.impl("project2d", &project2d_cuda);
}

}