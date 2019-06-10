#pragma once

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/float16.h"
#include "chainerx/macro.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {

class CudnnError : public ChainerxError {
public:
    using ChainerxError::ChainerxError;

    explicit CudnnError(cudnnStatus_t status);
    cudnnStatus_t error() const noexcept { return status_; }

private:
    cudnnStatus_t status_;
};

void CheckCudnnError(cudnnStatus_t status);

namespace cuda_internal {

// Returns a pointer to a cuDNN coefficient value of given type, allocated on the static storage.
template <int kValue>
const void* GetCudnnCoefficientPtr(Dtype dtype) {
    // TODO(niboshi): Get rid of the assumption that native and cuda float16 share the same representation.
    static const float kFloat32Value{kValue};
    static const double kFloat64Value{kValue};

    switch (dtype) {
        case Dtype::kFloat16:
            // fallthrough: cuDNN accepts float32 coefficients for float16 tensor operations.
        case Dtype::kFloat32:
            return &kFloat32Value;
        case Dtype::kFloat64:
            return &kFloat64Value;
        default:
            CHAINERX_NEVER_REACH();
    }
}

class CudnnTensorDescriptor {
public:
    explicit CudnnTensorDescriptor(const Array& arr);
    ~CudnnTensorDescriptor();

    cudnnTensorDescriptor_t descriptor() const { return desc_; }
    cudnnTensorDescriptor_t operator*() const { return desc_; }

private:
    CudnnTensorDescriptor();
    cudnnTensorDescriptor_t desc_{};
};

class CudnnFilterDescriptor {
public:
    explicit CudnnFilterDescriptor(const Array& w);
    ~CudnnFilterDescriptor();

    cudnnFilterDescriptor_t descriptor() const { return desc_; }
    cudnnFilterDescriptor_t operator*() const { return desc_; }

private:
    CudnnFilterDescriptor();
    cudnnFilterDescriptor_t desc_{};
};

class CudnnConvolutionDescriptor {
public:
    explicit CudnnConvolutionDescriptor(
            Dtype dtype,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride,
            const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
            int groups);
    ~CudnnConvolutionDescriptor();

    cudnnConvolutionDescriptor_t descriptor() const { return desc_; }
    cudnnConvolutionDescriptor_t operator*() const { return desc_; }

private:
    CudnnConvolutionDescriptor();
    cudnnConvolutionDescriptor_t desc_{};
};

class CudnnPoolingDescriptor {
public:
    explicit CudnnPoolingDescriptor(
            cudnnPoolingMode_t mode,
            cudnnNanPropagation_t max_pooling_nan_opt,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);
    ~CudnnPoolingDescriptor();

    cudnnPoolingDescriptor_t descriptor() const { return desc_; }
    cudnnPoolingDescriptor_t operator*() const { return desc_; }

private:
    CudnnPoolingDescriptor();
    cudnnPoolingDescriptor_t desc_{};
};

// cuDNN API calls using same handle is not thread-safe.
// This class ensures that the API calls are serialized using mutex lock.
class CudnnHandle {
public:
    explicit CudnnHandle(int device_index) : device_index_{device_index} {}
    ~CudnnHandle();

    template <class Func, class... Args>
    void Call(Func&& func, Args&&... args) {
        std::lock_guard<std::mutex> lock{handle_mutex_};
        CheckCudnnError(func(handle(), args...));
    }

private:
    cudnnHandle_t handle();

    int device_index_;
    std::mutex handle_mutex_{};
    cudnnHandle_t handle_{};
};

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
