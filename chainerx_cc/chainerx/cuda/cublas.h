#pragma once

#include <cublas_v2.h>

#include "chainerx/error.h"

namespace chainerx {
namespace cuda {

class CublasError : public ChainerxError {
public:
    explicit CublasError(cublasStatus_t status);
    cublasStatus_t error() const noexcept { return status_; }

private:
    cublasStatus_t status_;
};

void CheckCublasError(cublasStatus_t status);

}  // namespace cuda
}  // namespace chainerx
