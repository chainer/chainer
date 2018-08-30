#include "xchainer/cuda/cublas.h"

#include <cublas_v2.h>

#include <string>

#include "xchainer/error.h"
#include "xchainer/macro.h"

namespace xchainer {
namespace cuda {
namespace {

std::string BuildErrorMessage(cublasStatus_t error) {
    switch (error) {
#define XCHAINER_MATCH_AND_RETURN_MSG(msg) \
    case msg:                              \
        return #msg

        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_SUCCESS);
        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_NOT_INITIALIZED);
        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_ALLOC_FAILED);
        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_INVALID_VALUE);
        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_ARCH_MISMATCH);
        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_MAPPING_ERROR);
        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_EXECUTION_FAILED);
        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_INTERNAL_ERROR);
        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_NOT_SUPPORTED);
        XCHAINER_MATCH_AND_RETURN_MSG(CUBLAS_STATUS_LICENSE_ERROR);

#undef XCHAINER_MATCH_AND_RETURN_MSG
    }
    XCHAINER_NEVER_REACH();
}

}  // namespace

CublasError::CublasError(cublasStatus_t status) : XchainerError{BuildErrorMessage(status)}, status_{status} {}

void CheckCublasError(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw CublasError{status};
    }
}

}  // namespace cuda
}  // namespace xchainer
