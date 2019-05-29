#pragma once

#include <mutex>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include "chainerx/error.h"

namespace chainerx {
namespace cuda {

void CheckCusolverError(cusolverStatus_t status);

namespace cuda_internal {

class CusolverHandle {
public:
    explicit CusolverHandle(int device_index) : device_index_{device_index} {}

    ~CusolverHandle();

    CusolverHandle(const CusolverHandle&) = delete;
    CusolverHandle(CusolverHandle&&) = delete;
    CusolverHandle& operator=(const CusolverHandle&) = delete;
    CusolverHandle& operator=(CusolverHandle&&) = delete;

    template <class Func, class... Args>
    void Call(Func&& func, Args&&... args) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCusolverError(func(handle(), args...));
    }

private:
    cusolverDnHandle_t handle();

    int device_index_;
    std::mutex handle_mutex_{};
    cusolverDnHandle_t handle_{};
};

}  // namespace cuda_internal

class CusolverError : public ChainerxError {
public:
    explicit CusolverError(cusolverStatus_t status);
    cusolverStatus_t error() const noexcept { return status_; }

private:
    cusolverStatus_t status_;
};

}  // namespace cuda
}  // namespace chainerx
