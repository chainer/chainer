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
void dsyevd_(
        char* jobz, char* uplo, int* n, double* a, int* lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int* info);
void ssyevd_(
        char* jobz, char* uplo, int* n, float* a, int* lda, float* w, float* work, int* lwork, int* iwork, int* liwork, int* info);
}
#endif  // CHAINERX_ENABLE_LAPACK

namespace chainerx {
namespace native {
namespace {

template <typename T>
void Syevd(char /*jobz*/, char /*uplo*/, int /*n*/, T* /*a*/, int /*lda*/, T* /*w*/, T* /*work*/, int /*lwork*/, int* /*iwork*/, int /*liwork*/, int* /*info*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by syevd (Eigen)"};
}

#if CHAINERX_ENABLE_LAPACK
template <>
void Syevd<double>(char jobz, char uplo, int n, double* a, int lda, double* w, double* work, int lwork, int* iwork, int liwork, int* info) {
    dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}

template <>
void Syevd<float>(char jobz, char uplo, int n, float* a, int lda, float* w, float* work, int lwork, int* iwork, int liwork, int* info) {
    ssyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork, info);
}
#endif  // CHAINERX_ENABLE_LAPACK

}  // namespace

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
                throw ChainerxError{"Unsuccessfull syevd (Eigen Decomposition) execution. Info = ", info};
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
