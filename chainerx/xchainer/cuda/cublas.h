#pragma once

#include <cublas_v2.h>

#include "xchainer/error.h"

namespace xchainer {
namespace cuda {

class CublasError : public XchainerError {
public:
    explicit CublasError(cublasStatus_t status);
    cublasStatus_t error() const noexcept { return status_; }

private:
    cublasStatus_t status_;
};

void CheckCublasError(cublasStatus_t status);

}  // namespace cuda
}  // namespace xchainer
