#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

#ifdef CHAINERX_ENABLE_BLAS
#include <chainerx/native/native_device/cblas.h>
#endif  // CHAINERX_ENABLE_BLAS

#include "chainerx/array.h"
#include "chainerx/backend.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/indexable_array.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/macro.h"
#include "chainerx/native/data_type.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/kernel_regist.h"
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
        return AsContiguous(a);
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
        out.device().backend().CallKernel<CopyKernel>(out_contiguous, out);
    }
}

}  // namespace
#endif  // CHAINERX_ENABLE_BLAS

namespace {

template <typename T>
T MultiplyAdd(T x, T y, T z) {
    return x * y + z;
}

bool MultiplyAdd(bool x, bool y, bool z) { return (x && y) || z; }

float MultiplyAdd(Float16 x, Float16 y, float z) { return std::fmaf(static_cast<float>(x), static_cast<float>(y), z); }

float MultiplyAdd(float x, float y, float z) { return std::fmaf(x, y, z); }

double MultiplyAdd(double x, double y, double z) { return std::fma(x, y, z); }

}  // namespace

class NativeDotKernel : public DotKernel {
public:
    void Call(const Array& a, const Array& b, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, b, out);

        // TODO(sonots): Support ndim >= 2
        if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) {
            throw DimensionError{"ChainerX dot supports only 2-dimensional arrays."};
        }

        if (out.GetTotalSize() == 0) {
            return;
        }

#ifdef CHAINERX_ENABLE_BLAS
        if (out.dtype() == Dtype::kFloat32 || out.dtype() == Dtype::kFloat64) {
            Gemm(a.dtype() == out.dtype() ? a : a.AsType(out.dtype()), b.dtype() == out.dtype() ? b : b.AsType(out.dtype()), out);
            return;
        }

        if (out.dtype() == Dtype::kFloat16) {
            Array a32 = a.AsType(Dtype::kFloat32, false);
            Array b32 = b.AsType(Dtype::kFloat32, false);
            Array acc = out.AsType(Dtype::kFloat32);
            Gemm(a32, b32, acc);

            // TODO(gwtnb): Replace Fill(0) and += with CopyTo when CopyTo is added
            out.Fill(0);
            out += acc.AsType(Dtype::kFloat16);
            return;
        }
#endif  // CHAINERX_ENABLE_BLAS

        const Array& a_cast = a.dtype() == out.dtype() ? a : a.AsType(out.dtype());
        const Array& b_cast = b.dtype() == out.dtype() ? b : b.AsType(out.dtype());

        out.Fill(0);
        VisitDtype(out.dtype(), [&](auto pt) {
            CHAINERX_ASSERT(a_cast.dtype() == out.dtype());
            CHAINERX_ASSERT(b_cast.dtype() == out.dtype());

            using T = typename decltype(pt)::type;

            IndexableArray<const T, 2> a_cast_iarray{a_cast};
            IndexableArray<const T, 2> b_cast_iarray{b_cast};
            IndexableArray<T, 2> out_iarray{out};

            int64_t m = a_cast.shape()[0];
            int64_t k = a_cast.shape()[1];
            int64_t n = b_cast.shape()[1];
            CHAINERX_ASSERT(b_cast.shape()[0] == k);
            CHAINERX_ASSERT(out.shape()[0] == m);
            CHAINERX_ASSERT(out.shape()[1] == n);

            using AccT = std::conditional_t<std::is_same<T, Float16>{}, float, T>;
            constexpr auto acc_dtype = PrimitiveType<AccT>::kDtype;

            Array acc = out.AsType(acc_dtype, false);
            IndexableArray<AccT, 2> acc_iarray{acc};
            for (int64_t i = 0; i < m; ++i) {
                for (int64_t l = 0; l < k; ++l) {
                    int64_t a_i_l[] = {i, l};
                    T a_value = native_internal::StorageToDataType<const T>(a_cast_iarray[a_i_l]);
                    for (int64_t j = 0; j < n; ++j) {
                        int64_t acc_i_j[] = {i, j};
                        int64_t b_l_j[] = {l, j};
                        T b_value = native_internal::StorageToDataType<const T>(b_cast_iarray[b_l_j]);
                        AccT& acc_value = native_internal::StorageToDataType<AccT>(acc_iarray[acc_i_j]);
                        acc_value = MultiplyAdd(a_value, b_value, acc_value);
                    }
                }
            }
            if (!std::is_same<T, AccT>{}) {
                for (int64_t i = 0; i < m; ++i) {
                    for (int64_t j = 0; j < n; ++j) {
                        int64_t i_j[] = {i, j};
                        AccT acc_value = native_internal::StorageToDataType<AccT>(acc_iarray[i_j]);
                        T& out_value = native_internal::StorageToDataType<T>(out_iarray[i_j]);
                        out_value = static_cast<T>(acc_value);
                    }
                }
            }
        });
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(DotKernel, NativeDotKernel);

}  // namespace native
}  // namespace chainerx
