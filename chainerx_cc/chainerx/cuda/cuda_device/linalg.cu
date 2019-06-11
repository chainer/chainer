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

        int m = a.shape()[0];
        int n = a.shape()[1];
        int mn = std::min(m, n);

        Array Q = Empty(Shape({0}), dtype, device);
        Array R = a.Transpose().Copy(); // QR decomposition is done in-place
        Array tau = Empty(Shape({mn}), dtype, device);

        auto qr_impl = [&](auto pt, auto geqrf_bufferSize, auto orgqr_bufferSize, auto geqrf, auto orgqr) -> std::tuple<Array, Array> {
            using T = typename decltype(pt)::type;
            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            T* r_ptr = static_cast<T*>(internal::GetRawOffsetData(R));
            T* tau_ptr = static_cast<T*>(internal::GetRawOffsetData(tau));

            int *devInfo;
            CheckCudaError(cudaMalloc(&devInfo, sizeof(int)));

            int buffersize_geqrf = 0;
            device_internals.cusolver_handle().Call(
                geqrf_bufferSize,
                m, n, r_ptr, n, &buffersize_geqrf);

            Array work = Empty(Shape({buffersize_geqrf}), dtype, device);
            T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            device_internals.cusolver_handle().Call(
                geqrf,
                m, n, r_ptr, m, tau_ptr, work_ptr, buffersize_geqrf, devInfo);

            int devInfo_h = 0;
            CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessfull geqrf (QR) execution. Info = ", devInfo_h};
            }

            if (mode == QRMode::r) {
                R = R.At(std::vector<ArrayIndex>{Slice{}, Slice{0, mn}}).Transpose();  // R = R[:, range(0, mn)].T
                // R = triu(R);
                return std::make_tuple(std::move(Q), std::move(R));
            }

            if (mode == QRMode::raw) {
                return std::make_tuple(std::move(R), std::move(tau));
            }

            int mc;
            if (mode == QRMode::complete && m > n) {
                mc = m;
                Q = Empty(Shape({m, m}), dtype, device);
            } else {
                mc = mn;
                Q = Empty(Shape({n, m}), dtype, device);
            }

            Q.At(std::vector<ArrayIndex>{Slice{0, n}, Slice{}}) = R;  // Q[range(0, n), :] = R
            T* q_ptr = static_cast<T*>(internal::GetRawOffsetData(Q));

            int buffersize_orgqr = 0;
            device_internals.cusolver_handle().Call(
                orgqr_bufferSize,
                m, mc, mn, q_ptr, m, tau_ptr, &buffersize_orgqr);

            work = Empty(Shape({buffersize_orgqr}), dtype, device);

            device_internals.cusolver_handle().Call(
                orgqr,
                m, mc, mn, q_ptr, m, tau_ptr, work_ptr, buffersize_orgqr, devInfo);

            CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessfull orgqr (QR) execution. Info = ", devInfo_h};
            }

            Q = Q.At(std::vector<ArrayIndex>{Slice{0, mc}, Slice{}}).Transpose();  // Q = Q[range(0, mc), :].T
            R = R.At(std::vector<ArrayIndex>{Slice{}, Slice{0, mc}}).Transpose();  // R = R[:, range(0, mc)].T
            // R = triu(R);
            return std::make_tuple(std::move(Q), std::move(R));
        };

        if (a.dtype() == Dtype::kFloat32) {
            return qr_impl(PrimitiveType<float>{}, cusolverDnSgeqrf_bufferSize, cusolverDnSorgqr_bufferSize, cusolverDnSgeqrf, cusolverDnSorgqr);
        } else {
            CHAINERX_ASSERT(a.dtype() == Dtype::kFloat64);
            return qr_impl(PrimitiveType<double>{}, cusolverDnDgeqrf_bufferSize, cusolverDnDorgqr_bufferSize, cusolverDnDgeqrf, cusolverDnDorgqr);
        }

    }
};

CHAINERX_CUDA_REGISTER_KERNEL(QRKernel, CudaQRKernel);

}  // namespace cuda
}  // namespace chainerx
