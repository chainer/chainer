#include "chainerx/cuda/cusolver.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <string>

#include "chainerx/cuda/cuda_set_device_scope.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {

CusolverDnHandle::~CusolverDnHandle() {
    if (handle_ != nullptr) {
        // NOTE: CudaSetDeviceScope is not available because it may throw
        int orig_index{0};
        cudaGetDevice(&orig_index);
        cudaSetDevice(device_index_);
        cusolverDnDestroy(handle_);
        cudaSetDevice(orig_index);
    }
}

cusolverDnHandle_t CusolverDnHandle::handle() {
    if (handle_ == nullptr) {
        CudaSetDeviceScope scope{device_index_};
        CheckCusolverError(cusolverDnCreate(&handle_));
    }
    return handle_;
}

}  // namespace cuda_internal

namespace {

std::string BuildErrorMessage(cusolverStatus_t error) {
    switch (error) {
#define CHAINERX_MATCH_AND_RETURN_MSG(msg) \
    case msg:                              \
        return #msg

        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_SUCCESS);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_NOT_INITIALIZED);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_ALLOC_FAILED);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_INVALID_VALUE);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_ARCH_MISMATCH);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_MAPPING_ERROR);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_EXECUTION_FAILED);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_INTERNAL_ERROR);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_NOT_SUPPORTED);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_ZERO_PIVOT);
        CHAINERX_MATCH_AND_RETURN_MSG(CUSOLVER_STATUS_INVALID_LICENSE);

#undef CHAINERX_MATCH_AND_RETURN_MSG

        default:
            return "(UNKNOWN)";
    }
    CHAINERX_NEVER_REACH();
}

}  // namespace

CusolverError::CusolverError(cusolverStatus_t status) : ChainerxError{BuildErrorMessage(status)}, status_{status} {}

void CheckCusolverError(cusolverStatus_t status) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw CusolverError{status};
    }
}

}  // namespace cuda
}  // namespace chainerx
