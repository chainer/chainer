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

extern "C" void dpotrf_(char* uplo, int* n, double* a, int* lda, int* info);
extern "C" void spotrf_(char* uplo, int* n, float* a, int* lda, int* info);

namespace chainerx {
namespace native {

class NativeCholeskyKernel : public CholeskyKernel {
public:
    void Call(const Array& a, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(out.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

#ifndef CHAINERX_LAPACK_AVAILABLE
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_LAPACK_AVAILABLE

        // potrf (cholesky) stores result in-place, therefore copy ``a`` to ``out`` and then pass ``out`` to the routine
        device.backend().CallKernel<CopyKernel>(a, out);

        Array out_contiguous = AsContiguous(out);
        CHAINERX_ASSERT(a.dtype() == out_contiguous.dtype());

        auto cholesky_impl = [&](auto pt, auto potrf) {
            using T = typename decltype(pt)::type;

            // Note that LAPACK uses Fortran order.
            // To compute a lower triangular matrix L = cholesky(A), we use LAPACK to compute an upper triangular matrix U = cholesky(A).
            char uplo = 'U';

            T* out_ptr = static_cast<T*>(internal::GetRawOffsetData(out_contiguous));
            int N = a.shape()[0];

            int info;
            potrf(&uplo, &N, out_ptr, &N, &info);

            if (info != 0) {
                throw ChainerxError{"Unsuccessfull potrf (Cholesky) execution. Info = ", info};
            }
        };

        switch (a.dtype()) {
            case Dtype::kFloat16:
                throw DtypeError{"Half-precision (float16) is not supported by Cholesky decomposition"};
                break;
            case Dtype::kFloat32:
                cholesky_impl(PrimitiveType<float>{}, spotrf_);
                break;
            case Dtype::kFloat64:
                cholesky_impl(PrimitiveType<double>{}, dpotrf_);
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(CholeskyKernel, NativeCholeskyKernel);

}  // namespace native
}  // namespace chainerx
