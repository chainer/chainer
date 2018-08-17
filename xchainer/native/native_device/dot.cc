#include "xchainer/native/native_device.h"

#include <cassert>
#include <cstdint>

#ifdef XCHAINER_ENABLE_BLAS
#include <cblas.h>
#endif  // XCHAINER_ENABLE_BLAS

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/indexable_array.h"
#include "xchainer/native/elementwise.h"
#include "xchainer/routines/creation.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace native {

#ifdef XCHAINER_ENABLE_BLAS
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
        assert(a.ndim() == 2);
        // Row-major
        // Note that this condition is slightly relaxed than Array::IsContiguous() which requires
        // a.strides()[0] == a.item_size() * a.shape()[1]
        if (a.strides()[1] == a.item_size() && a.strides()[0] / a.item_size() >= a.shape()[1] && a.strides()[0] % a.item_size() == 0) {
            ld = a.strides()[0] / a.item_size();
            return a;
        }
        // Column-major
        if (a.strides()[0] == a.item_size() && a.strides()[1] / a.item_size() >= a.shape()[0] && a.strides()[1] % a.item_size() == 0) {
            ld = a.strides()[1] / a.item_size();
            trans = CblasTrans;
            return a;
        }
        // Force row-major contiguous
        ld = a.shape()[1];
        return AsContiguousArray(a);
    }
};

void Gemm(const Array& a, const Array& b, const Array& out) {
    assert(a.ndim() == 2);
    assert(b.ndim() == 2);
    assert(out.ndim() == 2);
    assert(out.dtype() == Dtype::kFloat32 || out.dtype() == Dtype::kFloat64);

    int64_t m = a.shape()[0];
    int64_t k = a.shape()[1];
    int64_t n = b.shape()[1];
    assert(b.shape()[0] == k);
    assert(out.shape()[0] == m);
    assert(out.shape()[1] == n);

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
        const T* a_ptr = internal::GetRawOffsetData<const T>(a_config);
        const T* b_ptr = internal::GetRawOffsetData<const T>(b_config);
        T* out_ptr = internal::GetRawOffsetData<T>(out_contiguous);
        GemmImpl<T>{}(
                CblasRowMajor, a_layout.trans, b_layout.trans, m, n, k, one, a_ptr, a_layout.ld, b_ptr, b_layout.ld, zero, out_ptr, n);
    };

    if (a.dtype() == Dtype::kFloat32) {
        gemm_impl(PrimitiveType<float>{});
    } else {
        assert(a.dtype() == Dtype::kFloat64);
        gemm_impl(PrimitiveType<double>{});
    }

    if (!is_out_contiguous) {
        out.device().Copy(out_contiguous, out);
    }
}

}  // namespace
#endif  // XCHAINER_ENABLE_BLAS

void NativeDevice::Dot(const Array& a, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, b, out);

    // TODO(sonots): Support ndim >= 2
    if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) {
        throw DimensionError{"XChainer dot supports only 2-dimensional arrays."};
    }

#ifdef XCHAINER_ENABLE_BLAS
    if (out.dtype() == Dtype::kFloat32 || out.dtype() == Dtype::kFloat64) {
        Gemm(a, b, out);
        return;
    }
#endif  // XCHAINER_ENABLE_BLAS

    out.Fill(0);
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T, 2> a_iarray{a};
        IndexableArray<const T, 2> b_iarray{b};
        IndexableArray<T, 2> out_iarray{out};

        int64_t m = a.shape()[0];
        int64_t k = a.shape()[1];
        int64_t n = b.shape()[1];
        assert(b.shape()[0] == k);
        assert(out.shape()[0] == m);
        assert(out.shape()[1] == n);

        for (int64_t i = 0; i < m; ++i) {
            for (int64_t l = 0; l < k; ++l) {
                int64_t a_i_l[] = {i, l};
                T a_value = a_iarray[a_i_l];
                for (int64_t j = 0; j < n; ++j) {
                    int64_t out_i_j[] = {i, j};
                    int64_t b_l_j[] = {l, j};
                    out_iarray[out_i_j] += a_value * b_iarray[b_l_j];
                }
            }
        }
    });
}

}  // namespace native
}  // namespace xchainer
