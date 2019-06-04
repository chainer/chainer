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
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/math.h"

namespace chainerx {
namespace cuda {

class CudaQRKernel : public QRKernel {
public:
    std::tuple<Array, Array> Call(const Array& a, QRMode mode = QRMode::reduced) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);

        throw NotImplementedError("QR decomposition is not yet implemented for cuda device");

        if (mode != QRMode::reduced) {
            throw NotImplementedError{"Modes other than reduce are not implemented yet"};
        }

        int m = a.shape()[0];
        int n = a.shape()[1];
        int lda = std::min(m, n);

        Array Q = Empty(Shape({m, n}), dtype, device);
        Array R = Empty(Shape({n, n}), dtype, device);
        Array tau = Empty(Shape({n, 1}), dtype, device);

        auto qr_impl = [&](auto pt, auto geqrf_bufferSize, auto orgqr_bufferSize, auto geqrf, auto orgqr) {
            using T = typename decltype(pt)::type;
            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            T* a_ptr = static_cast<T*>(internal::GetRawOffsetData(a));
            T* q_ptr = static_cast<T*>(internal::GetRawOffsetData(Q));
            T* r_ptr = static_cast<T*>(internal::GetRawOffsetData(R));
            T* tau_ptr = static_cast<T*>(internal::GetRawOffsetData(tau));

            int *devInfo;
            CheckCudaError(cudaMalloc(&devInfo, sizeof(int)));

            int lwork_geqrf = 0;
            int lwork_orgqr = 0;
            int lwork = 0;

            device_internals.cusolver_handle().Call(
                geqrf_bufferSize,
                m, n, a_ptr, lda, &lwork_geqrf);

            device_internals.cusolver_handle().Call(
                orgqr_bufferSize,
                m, n, n, a_ptr, lda, tau_ptr, &lwork_orgqr);

            lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;

            Array work = Empty(Shape({lwork, 1}), dtype, device);
            T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            device_internals.cusolver_handle().Call(
                geqrf,
                m, n, a_ptr, lda, tau_ptr, work_ptr, lwork, devInfo);

            int devInfo_h = 0;
            CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessfull geqrf (QR) execution. Info = ", devInfo_h};
            }

            device_internals.cusolver_handle().Call(
                orgqr,
                m, n, n, a_ptr, lda, tau_ptr, work_ptr, lwork, devInfo);

            CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessfull orgqr (QR) execution. Info = ", devInfo_h};
            }
        };

        if (a.dtype() == Dtype::kFloat32) {
            qr_impl(PrimitiveType<float>{}, cusolverDnSgeqrf_bufferSize, cusolverDnSorgqr_bufferSize, cusolverDnSgeqrf, cusolverDnSorgqr);
        } else {
            CHAINERX_ASSERT(a.dtype() == Dtype::kFloat64);
            qr_impl(PrimitiveType<double>{}, cusolverDnDgeqrf_bufferSize, cusolverDnDorgqr_bufferSize, cusolverDnDgeqrf, cusolverDnDorgqr);
        }

        return std::make_tuple(std::move(Q), std::move(R));

    }
};

CHAINERX_CUDA_REGISTER_KERNEL(QRKernel, CudaQRKernel);

}  // namespace cuda
}  // namespace chainerx
