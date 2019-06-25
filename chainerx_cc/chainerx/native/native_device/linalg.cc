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

// gesv
extern "C" void dgesv_(int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
extern "C" void sgesv_(int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info);

// getrf
extern "C" void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
extern "C" void sgetrf_(int* m, int* n, float* a, int* lda, int* ipiv, int* info);

// getri
extern "C" void dgetri_(int* n, double* a, int* lda, int* ipiv, double* work, int* lwork, int* info);
extern "C" void sgetri_(int* n, float* a, int* lda, int* ipiv, float* work, int* lwork, int* info);

namespace chainerx {
namespace native {

class NativeSolveKernel : public SolveKernel {
public:
    void Call(const Array& a, const Array& b, const Array& out) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

#ifndef CHAINERX_ENABLE_LAPACK
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_ENABLE_LAPACK

        auto solve_impl = [&](auto pt, auto gesv) {
            using T = typename decltype(pt)::type;

            Array lu_matrix = Empty(a.shape(), dtype, device);
            device.backend().CallKernel<CopyKernel>(a.Transpose(), lu_matrix);
            T* lu_ptr = static_cast<T*>(internal::GetRawOffsetData(lu_matrix));

            int n = a.shape()[0];
            int nrhs = 1;
            if (b.ndim() == 2) {
                nrhs = b.shape()[1];
            }

            Array ipiv = Empty(Shape({n}), Dtype::kInt32, device);
            int* ipiv_ptr = static_cast<int*>(internal::GetRawOffsetData(ipiv));

            device.backend().CallKernel<CopyKernel>(b, out);
            T* out_ptr = static_cast<T*>(internal::GetRawOffsetData(out));

            int info;
            gesv(&n, &nrhs, lu_ptr, &n, ipiv_ptr, out_ptr, &n, &info);

            if (info != 0) {
                throw ChainerxError{"Unsuccessfull gesv (Solve) execution. Info = ", info};
            }
        };

        switch (a.dtype()) {
            case Dtype::kFloat16:
                throw DtypeError{"Half-precision (float16) is not supported by solve"};
                break;
            case Dtype::kFloat32:
                solve_impl(PrimitiveType<float>{}, sgesv_);
                break;
            case Dtype::kFloat64:
                solve_impl(PrimitiveType<double>{}, dgesv_);
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(SolveKernel, NativeSolveKernel);

class NativeInverseKernel : public InverseKernel {
public:
    void Call(const Array& a, const Array& out) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

#ifndef CHAINERX_ENABLE_LAPACK
        throw ChainerxError{"LAPACK is not linked to ChainerX."};
#endif  // CHAINERX_ENABLE_LAPACK

        auto solve_impl = [&](auto pt, auto getrf, auto getri) {
            using T = typename decltype(pt)::type;

            device.backend().CallKernel<CopyKernel>(a, out);
            T* out_ptr = static_cast<T*>(internal::GetRawOffsetData(out));

            int n = a.shape()[0];

            Array ipiv = Empty(Shape({n}), Dtype::kInt32, device);
            int* ipiv_ptr = static_cast<int*>(internal::GetRawOffsetData(ipiv));

            int info;
            getrf(&n, &n, out_ptr, &n, ipiv_ptr, &info);
            if (info != 0) {
                throw ChainerxError{"Unsuccessfull getrf (LU) execution. Info = ", info};
            }

            int buffersize = -1;
            T work_size;
            getri(&n, out_ptr, &n, ipiv_ptr, &work_size, &buffersize, &info);
            buffersize = static_cast<int>(work_size);
            Array work = Empty(Shape({buffersize}), dtype, device);
            T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            getri(&n, out_ptr, &n, ipiv_ptr, work_ptr, &buffersize, &info);
            if (info != 0) {
                throw ChainerxError{"Unsuccessfull getri (Inverse LU) execution. Info = ", info};
            }
        };

        switch (a.dtype()) {
            case Dtype::kFloat16:
                throw DtypeError{"Half-precision (float16) is not supported by solve"};
                break;
            case Dtype::kFloat32:
                solve_impl(PrimitiveType<float>{}, sgetrf_, sgetri_);
                break;
            case Dtype::kFloat64:
                solve_impl(PrimitiveType<double>{}, dgetrf_, dgetri_);
                break;
            default:
                CHAINERX_NEVER_REACH();
        }
    }
};

CHAINERX_NATIVE_REGISTER_KERNEL(InverseKernel, NativeInverseKernel);

}  // namespace native
}  // namespace chainerx
