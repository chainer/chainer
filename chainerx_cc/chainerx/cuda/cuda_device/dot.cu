#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"

namespace chainerx {
namespace cuda {
namespace {

// Dispatch gemm routines based on the element type T
template <typename T>
struct Gemm;

template <>
struct Gemm<float> {
    template <typename... Args>
    void operator()(Args&&... args) const {
        CheckCublasError(cublasSgemm(std::forward<Args>(args)...));
    }
};

template <>
struct Gemm<double> {
    template <typename... Args>
    void operator()(Args&&... args) const {
        CheckCublasError(cublasDgemm(std::forward<Args>(args)...));
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

void CudaDevice::Dot(const Array& a, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, b, out);
    CudaSetDeviceScope scope{index()};

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
        Sum(a.Reshape({k}) * b.Reshape({k}), {0}, out.Reshape({}));
        return;
    }

    if (out.dtype() == Dtype::kFloat16) {
        // TODO(imanishi): Use cublasHgemm
        Array out_float32 = Empty(out.shape(), Dtype::kFloat32, *this);
        Dot(a.AsType(Dtype::kFloat32), b.AsType(Dtype::kFloat32), out_float32);
        AsType(out_float32, out);
        return;
    }

    bool is_out_contiguous = out.IsContiguous();
    Array out_contiguous = is_out_contiguous ? out : EmptyLike(out, *this);

    auto gemm_impl = [&](auto pt) {
        using T = typename decltype(pt)::type;
        using StorageType = cuda_internal::StorageType<T>;
        using CudaType = cuda_internal::DataType<T>;

        // Note that cuBLAS uses Fortran order.
        // To compute out = a x b, we use cuBLAS to compute out^T = b^T x a^T (here x is the matrix product).

        GemmInputLayout a_layout;
        GemmInputLayout b_layout;
        Array a_config = a_layout.Configure(a);
        Array b_config = b_layout.Configure(b);

        const CudaType one{chainerx::Float16{1}};
        const CudaType zero{chainerx::Float16{0}};
        const CudaType* a_ptr =
                &cuda_internal::StorageToDataType<const T>(*static_cast<const StorageType*>(internal::GetRawOffsetData(a_config)));
        const CudaType* b_ptr =
                &cuda_internal::StorageToDataType<const T>(*static_cast<const StorageType*>(internal::GetRawOffsetData(b_config)));
        CudaType* out_ptr = &cuda_internal::StorageToDataType<T>(*static_cast<StorageType*>(internal::GetRawOffsetData(out_contiguous)));

        std::lock_guard<std::mutex> lock{cublas_handle_mutex_};
        Gemm<T>{}(
                cublas_handle(), b_layout.trans, a_layout.trans, n, m, k, &one, b_ptr, b_layout.ld, a_ptr, a_layout.ld, &zero, out_ptr, n);
    };

    if (a.dtype() == Dtype::kFloat32) {
        gemm_impl(PrimitiveType<float>{});
    } else if (a.dtype() == Dtype::kFloat64) {
        gemm_impl(PrimitiveType<double>{});
    } else {
        throw NotImplementedError("dot is not implemented for non-float types in CUDA");
    }

    if (!is_out_contiguous) {
        Copy(out_contiguous, out);
    }
}

}  // namespace cuda
}  // namespace chainerx
