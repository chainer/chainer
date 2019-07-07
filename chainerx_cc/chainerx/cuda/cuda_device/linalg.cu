#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/cusolver.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/native/native_device.h"
#include "chainerx/routines/creation.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
cusolverStatus_t PotrfBuffersize(
        cusolverDnHandle_t /*handle*/, cublasFillMode_t /*uplo*/, int /*n*/, T* /*a*/, int /*lda*/, int* /*lwork*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by potrf (Cholesky)"};
}

template <typename T>
cusolverStatus_t Potrf(
        cusolverDnHandle_t /*handle*/,
        cublasFillMode_t /*uplo*/,
        int /*n*/,
        T* /*a*/,
        int /*lda*/,
        T* /*workspace*/,
        int /*lwork*/,
        int* /*devinfo*/) {
    throw DtypeError{"Only Arrays of float or double type are supported by potrf (Cholesky)"};
}

template <>
cusolverStatus_t PotrfBuffersize<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* a, int lda, int* lwork) {
    return cusolverDnDpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
}

template <>
cusolverStatus_t PotrfBuffersize<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* a, int lda, int* lwork) {
    return cusolverDnSpotrf_bufferSize(handle, uplo, n, a, lda, lwork);
}

template <>
cusolverStatus_t Potrf<double>(
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* a, int lda, double* workspace, int lwork, int* devinfo) {
    return cusolverDnDpotrf(handle, uplo, n, a, lda, workspace, lwork, devinfo);
}

template <>
cusolverStatus_t Potrf<float>(
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* a, int lda, float* workspace, int lwork, int* devinfo) {
    return cusolverDnSpotrf(handle, uplo, n, a, lda, workspace, lwork, devinfo);
}

}  // namespace

class CudaCholeskyKernel : public CholeskyKernel {
public:
    void Call(const Array& a, const Array& out) override {
        Device& device = a.device();
        device.CheckDevicesCompatible(a, out);
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);
        CHAINERX_ASSERT(out.ndim() == 2);
        CHAINERX_ASSERT(a.shape()[0] == a.shape()[1]);

        // potrf (cholesky) stores result in-place, therefore copy ``a`` to ``out`` and then pass ``out`` to the routine
        device.backend().CallKernel<CopyKernel>(a, out);

        Array out_contiguous = AsContiguous(out);

        auto cholesky_impl = [&](auto pt) {
            CHAINERX_ASSERT(a.dtype() == out_contiguous.dtype());

            using T = typename decltype(pt)::type;

            // Note that cuSOLVER uses Fortran order.
            // To compute a lower triangular matrix L = cholesky(A), we use cuSOLVER to compute an upper triangular matrix U = cholesky(A).
            cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

            cuda_internal::DeviceInternals& device_internals = cuda_internal::GetDeviceInternals(static_cast<CudaDevice&>(device));

            // compute workspace size and prepare workspace
            T* out_ptr = static_cast<T*>(internal::GetRawOffsetData(out_contiguous));
            int work_size = 0;
            const int N = a.shape()[0];
            device_internals.cusolverdn_handle().Call(PotrfBuffersize<T>, uplo, N, out_ptr, N, &work_size);

            // POTRF execution
            Array work = Empty(Shape({work_size}), dtype, device);
            T* work_ptr = static_cast<T*>(internal::GetRawOffsetData(work));

            std::shared_ptr<void> devInfo = device.Allocate(sizeof(int));
            device_internals.cusolverdn_handle().Call(Potrf<T>, uplo, N, out_ptr, N, work_ptr, work_size, static_cast<int*>(devInfo.get()));

            int devInfo_h = 0;
            Device& native_device = dynamic_cast<native::NativeDevice&>(GetDefaultContext().GetDevice({"native", 0}));
            device.MemoryCopyTo(&devInfo_h, devInfo.get(), sizeof(int), native_device);
            if (devInfo_h != 0) {
                throw ChainerxError{"Unsuccessfull potrf (Cholesky) execution. Info = ", devInfo_h};
            }
        };

        VisitFloatingPointDtype(dtype, cholesky_impl);
    }
};

CHAINERX_CUDA_REGISTER_KERNEL(CholeskyKernel, CudaCholeskyKernel);

}  // namespace cuda
}  // namespace chainerx
