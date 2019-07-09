#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

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
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/shape.h"

#if CHAINERX_ENABLE_LAPACK
extern "C" {
// gesdd
void dgesdd_(
    char* jobz,
    int* m,
    int* n,
    double* a,
    int* lda,
    double* s,
    double* u,
    int* ldu,
    double* vt,
    int* ldvt,
    double* work,
    int* lwork,
    int* iwork,
    int* info);
void sgesdd_(
    char* jobz,
    int* m,
    int* n,
    float* a,
    int* lda,
    float* s,
    float* u,
    int* ldu,
    float* vt,
    int* ldvt,
    float* work,
    int* lwork,
    int* iwork,
    int* info);
}
#endif  // CHAINERX_ENABLE_LAPACK

namespace chainerx {
namespace native {

class NativeSVDKernel : public SVDKernel {
public:
    std::tuple<Array, Array, Array> Call(const Array& a, bool full_matrices, bool compute_uv) override {
#if CHAINERX_ENABLE_LAPACK
        Device& device = a.device();
        Dtype dtype = a.dtype();

        CHAINERX_ASSERT(a.ndim() == 2);

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

        auto svd_impl = [&](auto pt, auto gesdd) -> std::tuple<Array, Array, Array> {
            using T = typename decltype(pt)::type;

            T* x_ptr = static_cast<T*>(internal::GetRawOffsetData(x));
            T* s_ptr = static_cast<T*>(internal::GetRawOffsetData(s));
            T* u_ptr = static_cast<T*>(internal::GetRawOffsetData(u));
            T* vt_ptr = static_cast<T*>(internal::GetRawOffsetData(vt));

            char job;
            if (compute_uv) {
                job = full_matrices ? 'A' : 'S';
            } else {
                job = 'N';
            }

            int info;
            int buffersize = -1;
            T work_size;
            gesdd(&job, &m, &n, x_ptr, &m, s_ptr, u_ptr, &m, vt_ptr, &n, &work_size, &buffersize, nullptr, &info);
            buffersize = static_cast<int>(work_size);

            Array work = Empty(Shape({buffersize}), dtype, device);
            T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            gesdd(&job, &m, &n, x_ptr, &m, s_ptr, u_ptr, &m, vt_ptr, &n, work_ptr, &buffersize, nullptr, &info);

            if (info != 0) {
                throw ChainerxError{"Unsuccessfull gesdd (SVD) execution. Info = ", info};
            }

            if (trans_flag) {
                return std::make_tuple(std::move(u.Transpose()), std::move(s), std::move(vt.Transpose()));
            } else {
                return std::make_tuple(std::move(vt), std::move(s), std::move(u));
            }
        };

        switch (a.dtype()) {
            case Dtype::kFloat16:
                throw DtypeError{"Half-precision (float16) is not supported by SVD"};
                break;
            case Dtype::kFloat32:
                return svd_impl(PrimitiveType<float>{}, sgesdd_);
                break;
            case Dtype::kFloat64:
                return svd_impl(PrimitiveType<double>{}, dgesdd_);
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
#else  // CHAINERX_LAPACK_AVAILABLE
        (void)a;  // unused
        (void)full_matrices;  // unused
        (void)compute_uv;  // unused
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_LAPACK_AVAILABLE
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(SVDKernel, NativeSVDKernel);

class NativePseudoInverseKernel : public PseudoInverseKernel {
public:
    void Call(const Array& a, const Array& out, float rcond = 1e-15) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);

        CHAINERX_ASSERT(a.ndim() == 2);

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

CHAINERX_NATIVE_REGISTER_KERNEL(PseudoInverseKernel, NativePseudoInverseKernel);

}  // namespace native
}  // namespace chainerx
