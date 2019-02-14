#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#ifdef CHAINERX_ENABLE_BLAS
#include <cblas.h>
#endif  // CHAINERX_ENABLE_BLAS

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/routines/creation.h"
#include "chainerx/shape.h"

namespace chainerx {
namespace native {

#ifdef CHAINERX_ENABLE_BLAS
namespace {

// Dispatch gemm routines based on the element type T
template <typename T>
struct GemmImpl;

template <>
struct GemmImpl<float> {
    template <typename... Args>
    void operator()(Args&&... args) const {
        cblas_sgemm(std::forward<Args>(args)...);
    }
};

template <>
struct GemmImpl<double> {
    template <typename... Args>
    void operator()(Args&&... args) const {
        cblas_dgemm(std::forward<Args>(args)...);
    }
};

struct GemmInputLayout {
    int64_t ld = 0;
    CBLAS_TRANSPOSE trans = CblasNoTrans;

    // Configure leading dimension and transposition accordingly, and makes the array C contiguous if necessary
    Array Configure(const Array& a) {
        CHAINERX_ASSERT(a.ndim() == 2);
        // Row-major
        // Note that this condition is slightly relaxed than Array::IsContiguous() which requires
        // a.strides()[0] == a.GetItemSize() * a.shape()[1]
        if (a.strides()[1] == a.GetItemSize() && a.strides()[0] / a.GetItemSize() >= a.shape()[1] &&
            a.strides()[0] % a.GetItemSize() == 0) {
            ld = a.strides()[0] / a.GetItemSize();
            return a;
        }
        // Column-major
        if (a.strides()[0] == a.GetItemSize() && a.strides()[1] / a.GetItemSize() >= a.shape()[0] &&
            a.strides()[1] % a.GetItemSize() == 0) {
            ld = a.strides()[1] / a.GetItemSize();
            trans = CblasTrans;
            return a;
        }
        // Force row-major contiguous
        ld = a.shape()[1];
        return internal::AsContiguous(a);
    }
};

void Gemm(const Array& a, const Array& b, const Array& out) {
    CHAINERX_ASSERT(a.ndim() == 2);
    CHAINERX_ASSERT(b.ndim() == 2);
    CHAINERX_ASSERT(out.ndim() == 2);
    CHAINERX_ASSERT(out.dtype() == Dtype::kFloat32 || out.dtype() == Dtype::kFloat64);

    int64_t m = a.shape()[0];
    int64_t k = a.shape()[1];
    int64_t n = b.shape()[1];
    CHAINERX_ASSERT(b.shape()[0] == k);
    CHAINERX_ASSERT(out.shape()[0] == m);
    CHAINERX_ASSERT(out.shape()[1] == n);

    bool is_out_contiguous = out.IsContiguous();
    Array out_contiguous = is_out_contiguous ? out : EmptyLike(out, out.device());

    auto gemm_impl = [&](auto pt) {
        using T = typename decltype(pt)::type;

        GemmInputLayout a_layout;
        GemmInputLayout b_layout;
        Array a_config = a_layout.Configure(a);
        Array b_config = b_layout.Configure(b);

        const T one = 1;
        const T zero = 0;
        const T* a_ptr = static_cast<const T*>(internal::GetRawOffsetData(a_config));
        const T* b_ptr = static_cast<const T*>(internal::GetRawOffsetData(b_config));
        T* out_ptr = static_cast<T*>(internal::GetRawOffsetData(out_contiguous));
        GemmImpl<T>{}(
                CblasRowMajor, a_layout.trans, b_layout.trans, m, n, k, one, a_ptr, a_layout.ld, b_ptr, b_layout.ld, zero, out_ptr, n);
    };

    if (a.dtype() == Dtype::kFloat32) {
        gemm_impl(PrimitiveType<float>{});
    } else {
        CHAINERX_ASSERT(a.dtype() == Dtype::kFloat64);
        gemm_impl(PrimitiveType<double>{});
    }

    if (!is_out_contiguous) {
        out.device().Copy(out_contiguous, out);
    }
}

}  // namespace
#endif  // CHAINERX_ENABLE_BLAS

namespace {

template <typename T>
T MultiplyAdd(T x, T y, T z) {
    return x * y + z;
}

Float16 MultiplyAdd(Float16 x, Float16 y, Float16 z) {
    return static_cast<Float16>(std::fmaf(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)));
}

float MultiplyAdd(float x, float y, float z) { return std::fmaf(x, y, z); }

double MultiplyAdd(double x, double y, double z) { return std::fma(x, y, z); }

}  // namespace

void NativeDevice::Dot(const Array& a, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, b, out);

    // TODO(sonots): Support ndim >= 2
    if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) {
        throw DimensionError{"ChainerX dot supports only 2-dimensional arrays."};
    }

#ifdef CHAINERX_ENABLE_BLAS
    if (out.dtype() == Dtype::kFloat32 || out.dtype() == Dtype::kFloat64) {
        Gemm(a, b, out);
        return;
    }
#endif  // CHAINERX_ENABLE_BLAS

    out.Fill(0);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        IndexableArray<const T, 2> a_iarray{a};
        IndexableArray<const T, 2> b_iarray{b};
        IndexableArray<T, 2> out_iarray{out};

        int64_t m = a.shape()[0];
        int64_t k = a.shape()[1];
        int64_t n = b.shape()[1];
        CHAINERX_ASSERT(b.shape()[0] == k);
        CHAINERX_ASSERT(out.shape()[0] == m);
        CHAINERX_ASSERT(out.shape()[1] == n);

        for (int64_t i = 0; i < m; ++i) {
            for (int64_t l = 0; l < k; ++l) {
                int64_t a_i_l[] = {i, l};
                T a_value = native_internal::StorageToDataType<const T>(a_iarray[a_i_l]);
                for (int64_t j = 0; j < n; ++j) {
                    int64_t out_i_j[] = {i, j};
                    int64_t b_l_j[] = {l, j};
                    T b_value = native_internal::StorageToDataType<const T>(b_iarray[b_l_j]);
                    T& out_value = native_internal::StorageToDataType<T>(out_iarray[out_i_j]);
                    out_value = MultiplyAdd(a_value, b_value, out_value);
                }
            }
        }
    });
}

}  // namespace native
}  // namespace chainerx
