#include "chainerx/cuda/cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <string>

#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {

CublasHandle::~CublasHandle() {
    if (handle_ != nullptr) {
        // NOTE: CudaSetDeviceScope is not available because it may throw
        int orig_index{0};
        cudaGetDevice(&orig_index);
        cudaSetDevice(device_index_);
        cublasDestroy(handle_);
        cudaSetDevice(orig_index);
    }
}

cublasHandle_t CublasHandle::handle() {
    if (handle_ == nullptr) {
        CudaSetDeviceScope scope{device_index_};
        CheckCublasError(cublasCreate(&handle_));
    }
    return handle_;
}

}  // namespace cuda_internal

namespace {

std::string BuildErrorMessage(cublasStatus_t error) {
    switch (error) {
#define CHAINERX_MATCH_AND_RETURN_MSG(msg) \
    case msg:                              \
        return #msg

        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_SUCCESS);
        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_NOT_INITIALIZED);
        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_ALLOC_FAILED);
        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_INVALID_VALUE);
        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_ARCH_MISMATCH);
        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_MAPPING_ERROR);
        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_EXECUTION_FAILED);
        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_INTERNAL_ERROR);
        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_NOT_SUPPORTED);
        CHAINERX_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_LICENSE_ERROR);

#undef CHAINERX_MATCH_AND_RETURN_MSG
    }
    CHAINERX_NEVER_REACH();
}

}  // namespace

CublasError::CublasError(cublasStatus_t status) : ChainerxError{BuildErrorMessage(status)}, status_{status} {}

void CheckCublasError(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CublasError{status};
    }
}

}  // namespace cuda
}  // namespace chainerx
