#pragma once

#include <cublas_v2.h>
#include <cudnn.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/cuda/cuda_conv.h"
#include "chainerx/cuda/memory_pool.h"
#include "chainerx/device.h"
#include "chainerx/routines/pooling.h"
#include "chainerx/scalar.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {

class CudaConvTest;  // for unit-tests

// Keeps any memory from being freed before CUDA asynchronous operations are finished.
// Operations in this class are thread safe.
class MemoryKeeper {
public:
    ~MemoryKeeper();

    // Registers a pointer to a memory chunk.
    // The memory is only freed after all preceding CUDA operations in the stream are finished.
    // TODO(niboshi): Currently only the default stream is supported.
    void Add(cudaStream_t stream, std::shared_ptr<void> memory);

    // Checks for recorded events and frees the associated memories.
    void Collect();

private:
    std::mutex mutex_{};
    std::queue<std::pair<cudaEvent_t, std::shared_ptr<void>>> queue_{};
};

}  // namespace cuda_internal

class CudaDevice : public Device {
public:
    ~CudaDevice() override;

    cuda_internal::CudnnHandle& cudnn_handle() { return cudnn_handle_; }

    const std::shared_ptr<MemoryPool>& device_memory_pool() { return device_memory_pool_; }

    void Synchronize() override;

    // memory.cc

    std::shared_ptr<void> Allocate(size_t bytesize) override;

    std::shared_ptr<void> MakeDataFromForeignPointer(const std::shared_ptr<void>& data) override;

    void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) override;

    void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) override;

    std::shared_ptr<void> TransferDataFrom(
            Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    // fill.cu

    void Fill(const Array& out, Scalar value) override;

    void Arange(Scalar start, Scalar step, const Array& out) override;

    void Identity(const Array& out) override;

    void Eye(int64_t k, const Array& out) override;

    void Diagflat(const Array& v, int64_t k, const Array& out) override;

    void Linspace(double start, double stop, const Array& out) override;

    // arithmetic.cu

    void Add(const Array& x1, const Array& x2, const Array& out) override;
    void AddAS(const Array& x1, Scalar x2, const Array& out) override;

    void Subtract(const Array& x1, const Array& x2, const Array& out) override;
    void SubtractAS(const Array& x1, Scalar x2, const Array& out) override;

    void Multiply(const Array& x1, const Array& x2, const Array& out) override;
    void MultiplyAS(const Array& x1, Scalar x2, const Array& out) override;

    void FloorDivide(const Array& x1, const Array& x2, const Array& out) override;
    void FloorDivideAS(const Array& x1, Scalar x2, const Array& out) override;

    void Divide(const Array& x1, const Array& x2, const Array& out) override;
    void DivideAS(const Array& x1, Scalar x2, const Array& out) override;

    // reduction.cu

    void ArgMax(const Array& a, const Axes& axis, const Array& out) override;

    void Sum(const Array& a, const Axes& axis, const Array& out) override;
    void AMax(const Array& a, const Axes& axis, const Array& out) override;

    // copy.cu

    void Copy(const Array& a, const Array& out) override;

    void AsType(const Array& a, const Array& out) override;

    // comparison.cu

    void Equal(const Array& x1, const Array& x2, const Array& out) override;

    void NotEqual(const Array& x1, const Array& x2, const Array& out) override;

    void Greater(const Array& x1, const Array& x2, const Array& out) override;

    void GreaterEqual(const Array& x1, const Array& x2, const Array& out) override;

    void LogicalNot(const Array& x1, const Array& out) override;

    // activation.cu

    void IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override;

    void Tanh(const Array& x, const Array& out) override;

    // dot.cc

    void Dot(const Array& a, const Array& b, const Array& out) override;

    // exp_log.cu

    void Exp(const Array& x, const Array& out) override;
    void Log(const Array& x, const Array& out) override;

    // misc.cu

    void Sqrt(const Array& x, const Array& out) override;

    void IsNan(const Array& x, const Array& out) override;
    void IsInf(const Array& x, const Array& out) override;

    // indexing.cu

    void Take(const Array& a, const Array& indices, int8_t axis, const Array& out) override;

    void AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) override;

    // conv.cc

    Array Conv(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override;

    Array ConvGradWeight(
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override;

    Array ConvTranspose(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& out_size) override;

    // pool.cc

    std::unique_ptr<MaxPoolForwardBackward> GetMaxPoolForwardBackward(
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override;

    std::unique_ptr<AveragePoolForwardBackward> GetAveragePoolForwardBackward(
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            AveragePoolPadMode pad_mode) override;

    // batch_norm.cc

    std::unique_ptr<BatchNormForwardBackward> GetBatchNormForwardBackward(
            const Array& running_mean, const Array& running_var, Scalar eps, Scalar decay, const Axes& axis) override;

    Array FixedBatchNorm(
            const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const Axes& axis)
            override;

protected:
    CudaDevice(CudaBackend& backend, int index)
        : Device{backend, index},
          device_memory_pool_{std::make_shared<MemoryPool>(index, std::make_unique<DeviceMemoryAllocator>())},
          pinned_memory_pool_{std::make_shared<MemoryPool>(index, std::make_unique<PinnedMemoryAllocator>())},
          cudnn_handle_{index} {}

private:
    friend CudaDevice* cuda_internal::CreateDevice(CudaBackend&, int);

    friend class cuda_internal::CudaConvTest;  // for unit-tests

    cublasHandle_t cublas_handle();  // not thread-safe

    // Allocates pinned memory.
    // The pinned memory is used internally by the CUDA device for asynchronous memory transfer, i.e. cudaMemcpyAsync.
    std::shared_ptr<void> AllocatePinnedMemory(size_t bytesize);

    // Asynchronous transfer from host to this device, w.r.t. host, using temporary pinned memory.
    // The current device must be set to this device, prior to calling this function.
    void MemoryCopyFromHostAsync(void* dst, const void* src, size_t bytesize);

    std::shared_ptr<MemoryPool> device_memory_pool_;

    // TODO(hvy): Consider checking if pinned memory is available by querying canMapHostMemory.
    std::shared_ptr<MemoryPool> pinned_memory_pool_;

    // Memory keeper.
    cuda_internal::MemoryKeeper memory_keeper_{};

    std::mutex cublas_handle_mutex_;
    cublasHandle_t cublas_handle_{};

    cuda_internal::CudnnHandle cudnn_handle_;

    cuda_internal::CudaConv cuda_conv_{};
};

}  // namespace cuda
}  // namespace chainerx
