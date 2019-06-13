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

        if (a.ndim() != 2 || out.ndim() != 2) {
            throw DimensionError{"ChainerX Cholesky only supports 2-dimensional arrays."};
        }
        if (a.shape()[0] != a.shape()[1]) {
            throw DimensionError{"Matrix is not square."};
        }

        // potrf (cholesky) stores result in-place, therefore copy ``a`` to ``out`` and then pass ``out`` to the routine
        device.backend().CallKernel<CopyKernel>(a, out);

        bool is_out_contiguous = out.IsContiguous();
        Array out_contiguous = is_out_contiguous ? out : AsContiguous(out);

        auto cholesky_impl = [&](auto pt, auto bufsize_func, auto solver_func) {
            CHAINERX_ASSERT(a.dtype() == out_contiguous.dtype());

            using T = typename decltype(pt)::type;

            // Note that cuSOLVER uses Fortran order.
            // To compute a lower triangular matrix L = cholesky(A), we use cuSOLVER to compute an upper triangular matrix U = cholesky(A).
            cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            // compute workspace size and prepare workspace
            T* out_ptr = static_cast<T*>(internal::GetRawOffsetData(out_contiguous));
            int work_size = 0;
            const int N = a.shape()[0];
            device_internals.cusolverdn_handle().Call(
                bufsize_func,
                uplo,
                N,
                out_ptr,
                N,
                &work_size);

            // POTRF execution
            T* work_space;
            CheckCudaError(cudaMalloc(&work_space, work_size * sizeof(T)));
            int *devInfo;
            CheckCudaError(cudaMalloc(&devInfo, sizeof(int)));
            device_internals.cusolverdn_handle().Call(
                solver_func,
                uplo,
                N,
                out_ptr,
                N,
                work_space,
                work_size,
                devInfo);

            int devInfo_h = 0;
            CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessfull potrf (Cholesky) execution. Info = ", devInfo_h};
            }
        };

        if (a.dtype() == Dtype::kFloat32) {
            cholesky_impl(PrimitiveType<float>{}, cusolverDnSpotrf_bufferSize, cusolverDnSpotrf);
        } else {
            CHAINERX_ASSERT(a.dtype() == Dtype::kFloat64);
            cholesky_impl(PrimitiveType<double>{}, cusolverDnDpotrf_bufferSize, cusolverDnDpotrf);
        }

        if (!is_out_contiguous) {
            device.backend().CallKernel<CopyKernel>(out_contiguous, out);
        }

    }
};

CHAINERX_CUDA_REGISTER_KERNEL(CholeskyKernel, CudaCholeskyKernel);

}  // namespace cuda
}  // namespace chainerx
