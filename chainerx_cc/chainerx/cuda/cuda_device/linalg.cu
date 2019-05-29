#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cusolver.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/kernels/math.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/math.h"

namespace chainerx {
namespace cuda {

class CudaCholeskyKernel : public CholeskyKernel {
public:
    void Call(const Array& a, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(out.ndim() == 2);

        // potrf (cholesky) stores result in-place, therefore copy ``a`` to ``out`` and then pass ``out`` to the routine
        device.backend().CallKernel<CopyKernel>(a, out);

        bool is_out_contiguous = out.IsContiguous();
        Array out_contiguous = is_out_contiguous ? out : AsContiguous(out);

        cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

        // compute workspace size and prepare workspace
        double* out_ptr = static_cast<double*>(internal::GetRawOffsetData(out_contiguous));
        int work_size = 0;
        const int N = a.shape()[0];
        device_internals.cusolver_handle().Call(
            cusolverDnDpotrf_bufferSize,
            CUBLAS_FILL_MODE_UPPER,
            N,
            out_ptr,
            N,
            &work_size);

        // POTRF execution
        double *work_space;
        CheckCudaError(cudaMalloc(&work_space, work_size * sizeof(double)));
        int *devInfo;
        CheckCudaError(cudaMalloc(&devInfo, sizeof(int)));
        device_internals.cusolver_handle().Call(
            cusolverDnDpotrf,
            CUBLAS_FILL_MODE_UPPER,
            N,
            out_ptr,
            N,
            work_space,
            work_size,
            devInfo);

        int devInfo_h = 0;
        CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (devInfo_h != 0) {
            std::cout << "Unsuccessful potrf execution\n\n" << "devInfo = " << devInfo_h << "\n\n";
        }

        if (!is_out_contiguous) {
            device.backend().CallKernel<CopyKernel>(out_contiguous, out);
        }

    }
};

CHAINERX_CUDA_REGISTER_KERNEL(CholeskyKernel, CudaCholeskyKernel);

}  // namespace cuda
}  // namespace chainerx
