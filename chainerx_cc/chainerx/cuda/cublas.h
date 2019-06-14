#pragma once

#include <mutex>

#include <cublas_v2.h>

#include "chainerx/error.h"

namespace chainerx {
namespace cuda {

void CheckCublasError(cublasStatus_t status);

namespace cuda_internal {

class CublasHandle {
public:
    explicit CublasHandle(int device_index) : device_index_{device_index} {}

    ~CublasHandle();

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle(CublasHandle&&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    CublasHandle& operator=(CublasHandle&&) = delete;

    template <class Func, class... Args>
    void Call(Func&& func, Args&&... args) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCublasError(func(handle(), args...));
    }

private:
    cublasHandle_t handle();

    int device_index_;
    std::mutex handle_mutex_{};
    cublasHandle_t handle_{};
};

}  // namespace cuda_internal

class CublasError : public ChainerxError {
public:
    explicit CublasError(cublasStatus_t status);
    cublasStatus_t error() const noexcept { return status_; }

private:
    cublasStatus_t status_;
};

}  // namespace cuda
}  // namespace chainerx
