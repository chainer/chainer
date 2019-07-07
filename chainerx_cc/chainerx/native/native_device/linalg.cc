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
#include "chainerx/shape.h"

#if CHAINERX_ENABLE_LAPACK
extern "C" {
// potrf
void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);
void spotrf_(char* uplo, int* n, float* a, int* lda, int* info);
}
#endif  // CHAINERX_ENABLE_LAPACK

namespace chainerx {
namespace native {
namespace {

template <typename T>
void Potrf(char /*uplo*/, int /*n*/, T* /*a*/, int /*lda*/, int* /*info*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by potrf (Cholesky)"};
}

#if CHAINERX_ENABLE_LAPACK
template <>
void Potrf<double>(char uplo, int n, double* a, int lda, int* info) {
    dpotrf_(&uplo, &n, a, &lda, info);
}

template <>
void Potrf<float>(char uplo, int n, float* a, int lda, int* info) {
    spotrf_(&uplo, &n, a, &lda, info);
}
#endif  // CHAINERX_ENABLE_LAPACK

}  // namespace

class NativeCholeskyKernel : public CholeskyKernel {
public:
    void Call(const Array& a, const Array& out) override {
#if CHAINERX_ENABLE_LAPACK
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(out.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

        // potrf (cholesky) stores result in-place, therefore copy ``a`` to ``out`` and then pass ``out`` to the routine
        device.backend().CallKernel<CopyKernel>(a, out);

        Array out_contiguous = AsContiguous(out);
        CHAINERX_ASSERT(a.dtype() == out_contiguous.dtype());

        auto cholesky_impl = [&](auto pt) {
            using T = typename decltype(pt)::type;

            // Note that LAPACK uses Fortran order.
            // To compute a lower triangular matrix L = cholesky(A), we use LAPACK to compute an upper triangular matrix U = cholesky(A).
            char uplo = 'U';

            T* out_ptr = static_cast<T*>(internal::GetRawOffsetData(out_contiguous));
            int N = a.shape()[0];

            int info;
            Potrf<T>(uplo, N, out_ptr, N, &info);

            if (info != 0) {
                throw ChainerxError{"Unsuccessfull potrf (Cholesky) execution. Info = ", info};
            }
        };

        VisitFloatingPointDtype(a.dtype(), cholesky_impl);
#else  // CHAINERX_LAPACK_AVAILABLE
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_LAPACK_AVAILABLE
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(CholeskyKernel, NativeCholeskyKernel);

}  // namespace native
}  // namespace chainerx
