#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cublas.h"
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
namespace {

// Dispatch gemm routines based on the element type T
template <typename T>
struct Gemm;

template <>
struct Gemm<float> {
    template <typename... Args>
    cublasStatus_t operator()(Args&&... args) const {
        return cublasSgemm(std::forward<Args>(args)...);
    }
};

template <>
struct Gemm<double> {
    template <typename... Args>
    cublasStatus_t operator()(Args&&... args) const {
        return cublasDgemm(std::forward<Args>(args)...);
    }
};

struct GemmInputLayout {
    int64_t ld = 0;
    cublasOperation_t trans = CUBLAS_OP_T;

    // Configure leading dimension and transposition accordingly, and makes the array C contiguous if necessary.
    Array Configure(const Array& a) {
        CHAINERX_ASSERT(a.ndim() == 2);
        // Row-major
        // Note that this condition is slightly relaxed than Array::IsContiguous() which requires
        // a.strides()[0] == a.GetItemSize() * a.shape()[1]
        if (a.strides()[1] == a.GetItemSize() && a.strides()[0] / a.GetItemSize() >= a.shape()[1] &&
            a.strides()[0] % a.GetItemSize() == 0) {
            ld = a.strides()[0] / a.GetItemSize();
            trans = CUBLAS_OP_N;  // transposed
            return a;
        }
        // Column-major
        if (a.strides()[0] == a.GetItemSize() && a.strides()[1] / a.GetItemSize() >= a.shape()[0] &&
            a.strides()[1] % a.GetItemSize() == 0) {
            ld = a.strides()[1] / a.GetItemSize();
            return a;
        }
        // Force row-major contiguous
        ld = a.shape()[1];
        trans = CUBLAS_OP_N;  // transposed
        return internal::AsContiguous(a);
    }
};

}  // namespace

class CudaDotKernel : public DotKernel {
public:
    void Call(const Array& a, const Array& b, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, b, out);
        CudaSetDeviceScope scope{device.index()};

        if (GetKind(out.dtype()) != DtypeKind::kFloat) {
            throw NotImplementedError("dot is not implemented for non-float types in CUDA");
        }

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(b.ndim() == 2);
        CHAINERX_ASSERT(out.ndim() == 2);

        int64_t m = a.shape()[0];
        int64_t k = a.shape()[1];
        int64_t n = b.shape()[1];
        CHAINERX_ASSERT(b.shape()[0] == k);
        CHAINERX_ASSERT(out.shape()[0] == m);
        CHAINERX_ASSERT(out.shape()[1] == n);

        if (m == 1 && n == 1) {
            // TODO(beam2d): Write a custom reduction kernel.
            // TODO(hvy): Avoid unnecessary cast here when multiplication supports mixed dtypes.
            const Array& a_cast = a.dtype() == out.dtype() ? a : a.AsType(out.dtype());
            const Array& b_cast = b.dtype() == out.dtype() ? b : b.AsType(out.dtype());
            device.backend().CallKernel<SumKernel>(a_cast.Reshape({k}) * b_cast.Reshape({k}), Axes{0}, out.Reshape({}));
            return;
        }

        if (out.dtype() == Dtype::kFloat16) {
            // TODO(imanishi): Use cublasHgemm
            Array out_float32 = Empty(out.shape(), Dtype::kFloat32, device);
            device.backend().CallKernel<DotKernel>(a.AsType(Dtype::kFloat32), b.AsType(Dtype::kFloat32), out_float32);
            device.backend().CallKernel<AsTypeKernel>(out_float32, out);
            return;
        }

        bool is_out_contiguous = out.IsContiguous();
        Array out_contiguous = is_out_contiguous ? out : EmptyLike(out, device);

        const Array& a_cast = a.dtype() == out.dtype() ? a : a.AsType(out.dtype());
        const Array& b_cast = b.dtype() == out.dtype() ? b : b.AsType(out.dtype());

        auto gemm_impl = [&](auto pt) {
            CHAINERX_ASSERT(a_cast.dtype() == out_contiguous.dtype());
            CHAINERX_ASSERT(b_cast.dtype() == out_contiguous.dtype());

            using T = typename decltype(pt)::type;
            using StorageType = cuda_internal::StorageType<T>;
            using CudaType = cuda_internal::DataType<T>;

            // Note that cuBLAS uses Fortran order.
            // To compute out = a x b, we use cuBLAS to compute out^T = b^T x a^T (here x is the matrix product).

            GemmInputLayout a_cast_layout;
            GemmInputLayout b_cast_layout;
            Array a_cast_config = a_cast_layout.Configure(a_cast);
            Array b_cast_config = b_cast_layout.Configure(b_cast);

            const CudaType one{chainerx::Float16{1}};
            const CudaType zero{chainerx::Float16{0}};
            const CudaType* a_cast_ptr =
                    &cuda_internal::StorageToDataType<const T>(*static_cast<const StorageType*>(internal::GetRawOffsetData(a_cast_config)));
            const CudaType* b_cast_ptr =
                    &cuda_internal::StorageToDataType<const T>(*static_cast<const StorageType*>(internal::GetRawOffsetData(b_cast_config)));
            CudaType* out_ptr =
                    &cuda_internal::StorageToDataType<T>(*static_cast<StorageType*>(internal::GetRawOffsetData(out_contiguous)));

            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            device_internals.cublas_handle().Call(
                    Gemm<T>{},
                    b_cast_layout.trans,
                    a_cast_layout.trans,
                    n,
                    m,
                    k,
                    &one,
                    b_cast_ptr,
                    b_cast_layout.ld,
                    a_cast_ptr,
                    a_cast_layout.ld,
                    &zero,
                    out_ptr,
                    n);
        };

        switch (out.dtype()) {
            case Dtype::kFloat32:
                gemm_impl(PrimitiveType<float>{});
                break;
            case Dtype::kFloat64:
                gemm_impl(PrimitiveType<double>{});
                break;
            default:
                CHAINERX_NEVER_REACH();
        }

        if (!is_out_contiguous) {
            device.backend().CallKernel<CopyKernel>(out_contiguous, out);
        }
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(DotKernel, CudaDotKernel);

}  // namespace cuda
}  // namespace chainerx
