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
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/math.h"

namespace chainerx {
namespace cuda {

class CudaSVDKernel : public SVDKernel {
public:
    std::tuple<Array, Array, Array> Call(const Array& a, bool full_matrices, bool compute_uv) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        if (a.ndim() != 2) {
            throw DimensionError{"ChainerX SVD supports only 2-dimensional arrays."};
        }

        int n = a.shape()[0];
        int m = a.shape()[1];

        Array x{};
        bool trans_flag;

        if (m >= n) {
            x = Empty(Shape({n, m}), dtype, device);
            device.backend().CallKernel<CopyKernel>(a, x);
            trans_flag = false;
        } else {
            m = a.shape()[0];
            n = a.shape()[1];
            x = Empty(Shape({n, m}), dtype, device);
            device.backend().CallKernel<CopyKernel>(a.Transpose(), x);
            trans_flag = true;
        }
        int mn = std::min(m, n);

        Array u{};
        Array vt{};

        if (compute_uv) {
            if (full_matrices) {
                u = Empty(Shape({m, m}), dtype, device);
                vt = Empty(Shape({n, n}), dtype, device);
            } else {
                u = Empty(Shape({mn, m}), dtype, device);
                vt = Empty(Shape({mn, n}), dtype, device);
            }
        } else {
            u = Empty(Shape({0}), dtype, device);
            vt = Empty(Shape({0}), dtype, device);
        }

        Array s = Empty(Shape({mn}), dtype, device);

        auto svd_impl = [&](auto pt, auto gesvd_bufferSize, auto gesvd) -> std::tuple<Array, Array, Array> {
            using T = typename decltype(pt)::type;
            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            T* x_ptr = static_cast<T*>(internal::GetRawOffsetData(x));
            T* s_ptr = static_cast<T*>(internal::GetRawOffsetData(s));
            T* u_ptr = static_cast<T*>(internal::GetRawOffsetData(u));
            T* vt_ptr = static_cast<T*>(internal::GetRawOffsetData(vt));

            int *devInfo;
            CheckCudaError(cudaMalloc(&devInfo, sizeof(int)));

            int buffersize = 0;
            device_internals.cusolver_handle().Call(
                gesvd_bufferSize,
                m, n, &buffersize);

            Array work = Empty(Shape({buffersize}), dtype, device);
            T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            signed char job;
            if (compute_uv) {
                job = full_matrices ? 'A' : 'S';
            } else {
                job = 'N';
            }

            device_internals.cusolver_handle().Call(
                gesvd,
                job, job, m, n, x_ptr, m,
                s_ptr, u_ptr, m, vt_ptr, n,
                work_ptr, buffersize, nullptr, devInfo);

            int devInfo_h = 0;
            CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessfull gesvd (SVD) execution. Info = ", devInfo_h};
            }

            if (trans_flag) {
                return std::make_tuple(std::move(u.Transpose()), std::move(s), std::move(vt.Transpose()));
            } else {
                return std::make_tuple(std::move(vt), std::move(s), std::move(u));
            }
        };

        if (a.dtype() == Dtype::kFloat32) {
            return svd_impl(PrimitiveType<float>{}, cusolverDnSgesvd_bufferSize, cusolverDnSgesvd);
        } else {
            CHAINERX_ASSERT(a.dtype() == Dtype::kFloat64);
            return svd_impl(PrimitiveType<double>{}, cusolverDnDgesvd_bufferSize, cusolverDnDgesvd);
        }

    }
};

CHAINERX_CUDA_REGISTER_KERNEL(SVDKernel, CudaSVDKernel);

class CudaPseudoInverseKernel : public PseudoInverseKernel {
public:
    void Call(const Array& a, const Array& out, float rcond = 1e-15) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        if (a.ndim() != 2 || out.ndim() != 2) {
            throw DimensionError{"ChainerX pseudo-inverse supports only 2-dimensional arrays."};
        }

        Array u{};
        Array s{};
        Array vt{};

        std::tie(u, s, vt) = device.backend().CallKernel<SVDKernel>(a, false, true);

        Array cutoff = rcond * s.Max();
        Array cutoff_indices = s <= cutoff;

        Array sinv = 1.0 / s;
        sinv = Where(cutoff_indices, 0, sinv);

        std::vector<ArrayIndex> indices{Slice{}, NewAxis{}};

        device.backend().CallKernel<DotKernel>(vt.Transpose(), sinv.At(indices) * u.Transpose(), out);

    }
};

CHAINERX_CUDA_REGISTER_KERNEL(PseudoInverseKernel, CudaPseudoInverseKernel);

}  // namespace cuda
}  // namespace chainerx
