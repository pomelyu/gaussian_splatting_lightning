#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

extern "C" {
    PyMODINIT_FUNC PyInit__C(void) {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",
            NULL,
            -1,
            NULL,
        };
        return PyModule_Create(&module_def);
    }
}

namespace gs_kernels {

at::Tensor project2d_cpu(const at::Tensor& points_3d, const at::Tensor& proj_matrix, double eps) {
    int N = points_3d.sizes()[0];
    TORCH_CHECK(points_3d.sizes() == at::IntArrayRef({N, 3}));
    TORCH_CHECK(points_3d.dtype() == at::kFloat);
    TORCH_CHECK(proj_matrix.sizes() == at::IntArrayRef({4, 4}));
    TORCH_CHECK(proj_matrix.dtype() == at::kFloat);
    TORCH_INTERNAL_ASSERT(points_3d.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(proj_matrix.device().type() == at::DeviceType::CPU);

    at::Tensor pt_contig = points_3d.contiguous();
    at::Tensor m_contig = proj_matrix.contiguous();
    at::Tensor result = at::empty(at::IntArrayRef({N, 2}), pt_contig.options());

    const float* pt_ptr = pt_contig.data_ptr<float>();
    const float* m_ptr = m_contig.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();

    for (int i = 0; i < N; i++) {
        float z = pt_ptr[i*3] * m_ptr[3] + pt_ptr[i*3 + 1] * m_ptr[7] + pt_ptr[i*3 + 2] * m_ptr[11] + m_ptr[15] + eps;
        float x = pt_ptr[i*3] * m_ptr[0] + pt_ptr[i*3 + 1] * m_ptr[4] + pt_ptr[i*3 + 2] * m_ptr[8] + m_ptr[12];
        float y = pt_ptr[i*3] * m_ptr[1] + pt_ptr[i*3 + 1] * m_ptr[5] + pt_ptr[i*3 + 2] * m_ptr[9] + m_ptr[13];
        result_ptr[i*2] = x / z;
        result_ptr[i*2+1] = y / z;
    }

    return result;
}

TORCH_LIBRARY(gs_kernels, m) {
    // Note that "float" in the schema corresponds to the C++ double type
    // and the Python float type.
    m.def("project2d(Tensor points_3d, Tensor proj_matrix, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(gs_kernels, CPU, m) {
    m.impl("project2d", &project2d_cpu);
}

}
