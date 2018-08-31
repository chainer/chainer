#include "chainerx/cuda/cublas.h"

#include <cublas_v2.h>

#include <string>

#include "chainerx/error.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {
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
