#pragma once

#include <mutex>

#include <cusolverDn.h>

#include "chainerx/error.h"

namespace chainerx {
namespace cuda {

void CheckCusolverError(cusolverStatus_t status);

namespace cuda_internal {

class CusolverDnHandle {
public:
    explicit CusolverDnHandle(int device_index) : device_index_{device_index} {}

    ~CusolverDnHandle();

    CusolverDnHandle(const CusolverDnHandle&) = delete;
    CusolverDnHandle(CusolverDnHandle&&) = delete;
    CusolverDnHandle& operator=(const CusolverDnHandle&) = delete;
    CusolverDnHandle& operator=(CusolverDnHandle&&) = delete;

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
