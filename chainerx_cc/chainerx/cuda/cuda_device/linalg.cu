#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cusolver.h"
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
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/math.h"

namespace chainerx {
namespace cuda {

class CudaSyevdKernel : public SyevdKernel {
public:
    std::tuple<Array, Array> Call(const Array& a, const std::string& UPLO, bool compute_eigen_vector) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);

        Array v = Empty(a.shape(), dtype, device);
        device.backend().CallKernel<CopyKernel>(a.Transpose(), v);

        int64_t m = a.shape()[0];
        int64_t lda = a.shape()[1];

        Array w = Empty(Shape({m}), dtype, device);

        auto syevd_impl = [&](auto pt, auto syevd_bufferSize, auto syevd) -> std::tuple<Array, Array> {
            using T = typename decltype(pt)::type;
            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            T* v_ptr = static_cast<T*>(internal::GetRawOffsetData(v));
            T* w_ptr = static_cast<T*>(internal::GetRawOffsetData(w));

            cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
            if (compute_eigen_vector) {
                jobz = CUSOLVER_EIG_MODE_VECTOR;
            }

            cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
            if (UPLO == "U") {
                uplo = CUBLAS_FILL_MODE_UPPER;
            }

            int buffersize = 0;
            device_internals.cusolverdn_handle().Call(syevd_bufferSize, jobz, uplo, m, v_ptr, lda, w_ptr, &buffersize);

            Array work = Empty(Shape({buffersize}), dtype, device);
            T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            int* devInfo;
            CheckCudaError(cudaMalloc(&devInfo, sizeof(int)));

            device_internals.cusolverdn_handle().Call(syevd, jobz, uplo, m, v_ptr, lda, w_ptr, work_ptr, buffersize, devInfo);

            int devInfo_h = 0;
            CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessfull syevd (Eigen Decomposition) execution. Info = ", devInfo_h};
            }

            return std::make_tuple(std::move(w), std::move(v.Transpose().Copy()));
        };

        switch (a.dtype()) {
            case Dtype::kFloat16:
                throw DtypeError{"Half-precision (float16) is not supported by eigen decomposition"};
                break;
            case Dtype::kFloat32:
                return syevd_impl(PrimitiveType<float>{}, cusolverDnSsyevd_bufferSize, cusolverDnSsyevd);
                break;
            case Dtype::kFloat64:
                return syevd_impl(PrimitiveType<double>{}, cusolverDnDsyevd_bufferSize, cusolverDnDsyevd);
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(SyevdKernel, CudaSyevdKernel);

}  // namespace cuda
}  // namespace chainerx
