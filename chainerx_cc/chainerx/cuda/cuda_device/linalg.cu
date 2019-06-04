#include "chainerx/cuda/cuda_device.h"

#include <cstdint>
#include <mutex>
#include <type_traits>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/backend.h"
#include "chainerx/backend_util.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cusolver.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/cuda/data_type.cuh"
#include "chainerx/cuda/float16.cuh"
#include "chainerx/cuda/kernel_regist.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/kernels/creation.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/kernels/math.h"
#include "chainerx/kernels/misc.h"
#include "chainerx/macro.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/math.h"

namespace chainerx {
namespace cuda {

class CudaQRKernel : public QRKernel {
public:
    std::tuple<Array, Array> Call(const Array& a, QRMode mode = QRMode::reduced) override {
        Device& device = a.device();
        Dtype dtype = a.dtype();
        CudaSetDeviceScope scope{device.index()};

        CHAINERX_ASSERT(a.ndim() == 2);

        throw NotImplementedError("QR decomposition is not yet implemented for cuda device");

        if (mode != QRMode::reduced) {
            throw NotImplementedError{"Modes other than reduce are not implemented"};
        }

        // using T = typename double;

        // const int m = a.shape[0];
        // const int n = a.shape[1];
        // const int lda = std::min(m, n);

        // Array Q = Empty(Shape(m, n), dtype, a.device());
        // Array R = Empty(Shape(n, n), dtype, a.device());

        // T d_tau;
        // CheckCudaError(cudaMalloc(&d_tau, n*sizeof(T)));
        // T d_R;
        // CheckCudaError(cudaMalloc(&d_R, n*n*sizeof(T)));
        // int *devInfo;
        // CheckCudaError(cudaMalloc(&devInfo, sizeof(int)));

        // int lwork_geqrf = 0;
        // int lwork_orgqr = 0;
        // int lwork = 0;
        // const float h_one = 1;
        // const float h_minus_one = -1;

        // device_internals.cusolver_handle().Call(
        //     cusolverDnDgeqrf_bufferSize,
        //     m, n, a_ptr, lda, &lwork_geqrf);
        
        // device_internals.cusolver_handle().Call(
        //     cusolverDnDorgqr_bufferSize,
        //     m, n, n, a_ptr, lda, d_tau, &lwork_orgqr);
        
        // lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;

        // T *d_work;
        // CheckCudaError(cudaMalloc(&d_work, lwork * sizeof(T)));

        // device_internals.cusolver_handle().Call(
        //     cusolverDnDgeqrf,
        //     m, n, a_ptr, lda, d_tau, d_work, lwork, devInfo);

        // int devInfo_h = 0;
        // CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        // if (devInfo_h != 0) {
        //     throw ChainerxError{"Unsuccessfull geqrf (QR) execution. Info = ", devInfo_h};
        // }

        // device_internals.cusolver_handle().Call(
        //     cusolverDnDorgqr,
        //     m, n, n, a_ptr, lda, d_tau, d_work, lwork, devInfo);
        
        // CheckCudaError(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
        // if (devInfo_h != 0) {
        //     throw ChainerxError{"Unsuccessfull orgqr (QR) execution. Info = ", devInfo_h};
        // }

    }
};

CHAINERX_CUDA_REGISTER_KERNEL(QRKernel, CudaQRKernel);

}  // namespace cuda
}  // namespace chainerx
