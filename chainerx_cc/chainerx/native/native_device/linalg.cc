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

// syevd
void dsyevd_(char* jobz, char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int* info);
void ssyevd_(char* jobz, char* uplo, int* n, float* a, int* lda, float* w, float* work, int* lwork, int* iwork, int* liwork, int* info);
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
void Syevd(
        char /*jobz*/,
        char /*uplo*/,
        int /*n*/,
        T* /*a*/,
        int /*lda*/,
        T* /*w*/,
        T* /*work*/,
        int /*lwork*/,
        int* /*iwork*/,
        int /*liwork*/,
        int* /*info*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by syevd (Eigen)"};
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
void Syevd<double>(char jobz, char uplo, int n, double* a, int lda, double* w, double* work, int lwork, int* iwork, int liwork, int* info) {
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template <>
void Syevd<float>(char jobz, char uplo, int n, float* a, int lda, float* w, float* work, int lwork, int* iwork, int liwork, int* info) {
    ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
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

        VisitFloatingPointDtype(a.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            SolveImpl<T>(a, b, out);
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

class NativeSyevdKernel : public SyevdKernel {
public:
    std::tuple<Array, Array> Call(const Array& a, const std::string& UPLO, bool compute_eigen_vector) override {
#if CHAINERX_ENABLE_LAPACK
        Device& device = a.device();
        Dtype dtype = a.dtype();

        CHAINERX_ASSERT(a.ndim() == 2);

        Array v = Empty(a.shape(), dtype, device);
        device.backend().CallKernel<CopyKernel>(a.Transpose(), v);

        int m = a.shape()[0];
        int lda = a.shape()[1];

        Array w = Empty(Shape{m}, dtype, device);

        auto syevd_impl = [&](auto pt) -> std::tuple<Array, Array> {
            using T = typename decltype(pt)::type;

            T* v_ptr = static_cast<T*>(internal::GetRawOffsetData(v));
            T* w_ptr = static_cast<T*>(internal::GetRawOffsetData(w));

            char jobz = 'N';
            if (compute_eigen_vector) {
                jobz = 'V';
            }

            char uplo = UPLO.c_str()[0];

            int info;
            int lwork = -1;
            int liwork = -1;
            T work_size;
            int iwork_size;

            Syevd<T>(jobz, uplo, m, v_ptr, lda, w_ptr, &work_size, lwork, &iwork_size, liwork, &info);

            lwork = static_cast<int>(work_size);
            Array work = Empty(Shape{lwork}, dtype, device);
            T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            liwork = static_cast<int>(iwork_size);
            Array iwork = Empty(Shape{liwork}, Dtype::kInt32, device);
            int* iwork_ptr = static_cast<int*>(internal::GetRawOffsetData(iwork));

            Syevd<T>(jobz, uplo, m, v_ptr, lda, w_ptr, work_ptr, lwork, iwork_ptr, liwork, &info);

            if (info != 0) {
                throw ChainerxError{"Unsuccessful syevd (Eigen Decomposition) execution. Info = ", info};
            }

            return std::make_tuple(std::move(w), std::move(v.Transpose().Copy()));
        };

        return VisitFloatingPointDtype(dtype, syevd_impl);
#else  // CHAINERX_LAPACK_AVAILABLE
        (void)a;
        (void)UPLO;
        (void)compute_eigen_vector;
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_LAPACK_AVAILABLE
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(SyevdKernel, NativeSyevdKernel);

}  // namespace native
}  // namespace chainerx
