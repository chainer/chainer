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
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/shape.h"

#if CHAINERX_ENABLE_LAPACK
extern "C" {
// gesv
void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
void sgesv_(int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);

// getrf
void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
void sgetrf_(int* m, int* n, float* a, int* lda, int* ipiv, int* info);

// getri
void dgetri_(int* n, double* a, int* lda, int* ipiv, double* work, int* lwork, int* info);
void sgetri_(int* n, float* a, int* lda, int* ipiv, float* work, int* lwork, int* info);

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
namespace {

template <typename T>
void Gesv(int /*n*/, int /*nrhs*/, T* /*a*/, int /*lda*/, int* /*ipiv*/, T* /*b*/, int /*ldb*/, int* /*info*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by gesv (Solve)"};
}

template <typename T>
void Getrf(int /*m*/, int /*n*/, T* /*a*/, int /*lda*/, int* /*ipiv*/, int* /*info*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by getri (LU)"};
}

template <typename T>
void Getri(int /*n*/, T* /*a*/, int /*lda*/, int* /*ipiv*/, T* /*work*/, int /*lwork*/, int* /*info*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by getri (Inverse LU)"};
}

template <typename T>
void Gesdd(
        char /*jobz*/,
        int /*m*/,
        int /*n*/,
        T* /*a*/,
        int /*lda*/,
        T* /*s*/,
        T* /*u*/,
        int /*ldu*/,
        T* /*vt*/,
        int /*ldvt*/,
        T* /*work*/,
        int /*lwork*/,
        int* /*iwork*/,
        int* /*info*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by gesdd (SVD)"};
}

#if CHAINERX_ENABLE_LAPACK
template <>
void Gesv<double>(int n, int nrhs, double* a, int lda, int* ipiv, double* b, int ldb, int* info) {
    dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void Gesv<float>(int n, int nrhs, float* a, int lda, int* ipiv, float* b, int ldb, int* info) {
    sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
}

template <>
void Getrf<double>(int m, int n, double* a, int lda, int* ipiv, int* info) {
    dgetrf_(&m, &n, a, &lda, ipiv, info);
}

template <>
void Getrf<float>(int m, int n, float* a, int lda, int* ipiv, int* info) {
    sgetrf_(&m, &n, a, &lda, ipiv, info);
}

template <>
void Getri<double>(int n, double* a, int lda, int* ipiv, double* work, int lwork, int* info) {
    dgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void Getri<float>(int n, float* a, int lda, int* ipiv, float* work, int lwork, int* info) {
    sgetri_(&n, a, &lda, ipiv, work, &lwork, info);
}

template <>
void Gesdd<double>(
        char jobz,
        int m,
        int n,
        double* a,
        int lda,
        double* s,
        double* u,
        int ldu,
        double* vt,
        int ldvt,
        double* work,
        int lwork,
        int* iwork,
        int* info) {
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info);
}

template <>
void Gesdd<float>(
        char jobz,
        int m,
        int n,
        float* a,
        int lda,
        float* s,
        float* u,
        int ldu,
        float* vt,
        int ldvt,
        float* work,
        int lwork,
        int* iwork,
        int* info) {
    sgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, info);
}
#endif  // CHAINERX_ENABLE_LAPACK

template <typename T>
void SolveImpl(const Array& a, const Array& b, const Array& out) {
    Device& device = a.device();
    Dtype dtype = a.dtype();

    Array lu_matrix = Empty(a.shape(), dtype, device);
    device.backend().CallKernel<CopyKernel>(a.Transpose(), lu_matrix);
    auto lu_ptr = static_cast<T*>(internal::GetRawOffsetData(lu_matrix));

    int64_t n = a.shape()[0];
    int64_t nrhs = 1;
    if (b.ndim() == 2) {
        nrhs = b.shape()[1];
    }

    Array ipiv = Empty(Shape{n}, Dtype::kInt32, device);
    auto ipiv_ptr = static_cast<int*>(internal::GetRawOffsetData(ipiv));

    Array out_transposed = b.Transpose().Copy();
    auto out_ptr = static_cast<T*>(internal::GetRawOffsetData(out_transposed));

    int info;
    Gesv(n, nrhs, lu_ptr, n, ipiv_ptr, out_ptr, n, &info);

    if (info != 0) {
        throw ChainerxError{"Unsuccessful gesv (Solve) execution. Info = ", info};
    }

    device.backend().CallKernel<CopyKernel>(out_transposed.Transpose(), out);
}

template <typename T>
void InverseImpl(const Array& a, const Array& out) {
    Device& device = a.device();
    Dtype dtype = a.dtype();

    device.backend().CallKernel<CopyKernel>(a, out);
    auto out_ptr = static_cast<T*>(internal::GetRawOffsetData(out));

    int64_t n = a.shape()[0];

    Array ipiv = Empty(Shape{n}, Dtype::kInt32, device);
    auto ipiv_ptr = static_cast<int*>(internal::GetRawOffsetData(ipiv));

    int info;
    Getrf(n, n, out_ptr, n, ipiv_ptr, &info);
    if (info != 0) {
        throw ChainerxError{"Unsuccessful getrf (LU) execution. Info = ", info};
    }

    int buffersize = -1;
    T work_size;
    Getri(n, out_ptr, n, ipiv_ptr, &work_size, buffersize, &info);
    buffersize = static_cast<int>(work_size);
    Array work = Empty(Shape{buffersize}, dtype, device);
    auto work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

    Getri(n, out_ptr, n, ipiv_ptr, work_ptr, buffersize, &info);
    if (info != 0) {
        throw ChainerxError{"Unsuccessful getri (Inverse LU) execution. Info = ", info};
    }
}

}  // namespace

class NativeSolveKernel : public SolveKernel {
public:
    void Call(const Array& a, const Array& b, const Array& out) override {
#if CHAINERX_ENABLE_LAPACK
        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            SolveImpl<T>(a.dtype() == out.dtype() ? a : a.AsType(out.dtype()), b.dtype() == out.dtype() ? b : b.AsType(out.dtype()), out);
        });
#else  // CHAINERX_ENABLE_LAPACK
        (void)a;  // unused
        (void)b;  // unused
        (void)out;  // unused
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_ENABLE_LAPACK
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(SolveKernel, NativeSolveKernel);

class NativeInverseKernel : public InverseKernel {
public:
    void Call(const Array& a, const Array& out) override {
#if CHAINERX_ENABLE_LAPACK

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

        VisitFloatingPointDtype(a.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            InverseImpl<T>(a, out);
        });
#else  // CHAINERX_ENABLE_LAPACK
        (void)a;  // unused
        (void)out;  // unused
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_ENABLE_LAPACK
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(InverseKernel, NativeInverseKernel);

class NativeSvdKernel : public SvdKernel {
public:
    void Call(const Array& a, const Array& u, const Array& s, const Array& vt, bool full_matrices) override {
#if CHAINERX_ENABLE_LAPACK
        Device& device = a.device();
        Dtype dtype = a.dtype();

        CHAINERX_ASSERT(a.ndim() == 2);

        bool compute_uv = u.shape()[0] != 0 && vt.shape()[0] != 0;

        // LAPACK assumes arrays are in column-major order.
        // In order to avoid transposing the input matrix, matrix dimensions are swapped.
        // Since the input is assumed to be transposed, it is necessary to
        // swap the pointers to u and vt matrices when calling Gesdd.
        int64_t n = a.shape()[0];
        int64_t m = a.shape()[1];
        int64_t k = std::min(m, n);
        int64_t ldu = m;
        int64_t ldvt = full_matrices ? n : k;

        Array x = EmptyLike(a, device);
        device.backend().CallKernel<CopyKernel>(a, x);

        auto svd_impl = [&](auto pt) {
            using T = typename decltype(pt)::type;

            auto x_ptr = static_cast<T*>(internal::GetRawOffsetData(x));
            auto s_ptr = static_cast<T*>(internal::GetRawOffsetData(s));
            auto u_ptr = static_cast<T*>(internal::GetRawOffsetData(u));
            auto vt_ptr = static_cast<T*>(internal::GetRawOffsetData(vt));

            char job;
            if (compute_uv) {
                job = full_matrices ? 'A' : 'S';
            } else {
                job = 'N';
            }

            Array iwork = Empty(Shape{8 * k}, Dtype::kInt64, device);
            auto iwork_ptr = static_cast<int*>(internal::GetRawOffsetData(iwork));

            int info;
            int buffersize = -1;
            T work_size;
            // When calling Gesdd pointers to u and vt are swapped instead of transposing the input matrix.
            Gesdd(job, m, n, x_ptr, m, s_ptr, vt_ptr, ldu, u_ptr, ldvt, &work_size, buffersize, iwork_ptr, &info);
            buffersize = static_cast<int>(work_size);

            Array work = Empty(Shape{buffersize}, dtype, device);
            auto work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            Gesdd(job, m, n, x_ptr, m, s_ptr, vt_ptr, ldu, u_ptr, ldvt, work_ptr, buffersize, iwork_ptr, &info);

            if (info != 0) {
                throw ChainerxError{"Unsuccessful gesdd (SVD) execution. Info = ", info};
            }
        };

        VisitFloatingPointDtype(dtype, svd_impl);
#else  // CHAINERX_LAPACK_AVAILABLE
        (void)a;  // unused
        (void)u;  // unused
        (void)s;  // unused
        (void)vt;  // unused
        (void)full_matrices;  // unused
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_LAPACK_AVAILABLE
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(SvdKernel, NativeSvdKernel);

}  // namespace native
}  // namespace chainerx
