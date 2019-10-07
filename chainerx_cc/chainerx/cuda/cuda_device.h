#pragma once

#include <cublas_v2.h>
#include <cudnn.h>
#include <cusolverDn.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/cuda/cuda_conv.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/cuda/cusolver.h"
#include "chainerx/cuda/memory_pool.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/kernels/normalization.h"
#include "chainerx/kernels/pooling.h"
#include "chainerx/kernels/rnn.h"
#include "chainerx/routines/normalization.h"
#include "chainerx/routines/pooling.h"
#include "chainerx/scalar.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {

class CudaDevice;

namespace cuda_internal {

class CudaConvTest;  // for unit-tests

// Keeps any memory from being freed before CUDA asynchronous operations are finished.
// Operations in this class are thread safe.
class MemoryKeeper {
public:
    MemoryKeeper() = default;

    ~MemoryKeeper();

    MemoryKeeper(const MemoryKeeper&) = delete;
    MemoryKeeper(MemoryKeeper&&) = delete;
    MemoryKeeper& operator=(const MemoryKeeper&) = delete;
    MemoryKeeper& operator=(MemoryKeeper&&) = delete;

    // Registers a pointer to a memory chunk.
    // The memory is only freed after all preceding CUDA operations in the stream are finished.
    // TODO(niboshi): Currently only the default stream is supported.
    void Add(cudaStream_t stream, std::shared_ptr<void> memory);

    // Checks for recorded events and frees the associated memories.
    void Collect();

private:
    std::mutex mutex_{};
    std::queue<std::pair<cudaEvent_t, std::shared_ptr<void>>> queue_{};
    std::atomic<bool> is_empty_{true};
};

// Keeps handles and other device internals.
// These internals are exposed through `GetDeviceInternals` for CUDA internal usages.
class DeviceInternals {
public:
    explicit DeviceInternals(int device_index)
        : cublas_handle_{device_index}, cudnn_handle_{device_index}, cusolverdn_handle_{device_index} {}

    ~DeviceInternals() = default;

    DeviceInternals(const DeviceInternals&) = delete;
    DeviceInternals(DeviceInternals&&) = delete;
    DeviceInternals& operator=(const DeviceInternals&) = delete;
    DeviceInternals& operator=(DeviceInternals&&) = delete;

    cuda_internal::CublasHandle& cublas_handle() { return cublas_handle_; }

    cuda_internal::CudnnHandle& cudnn_handle() { return cudnn_handle_; }

    cuda_internal::CusolverDnHandle& cusolverdn_handle() { return cusolverdn_handle_; }

    cuda_internal::CudaConv& cuda_conv() { return cuda_conv_; }

private:
    cuda_internal::CublasHandle cublas_handle_;

    cuda_internal::CudnnHandle cudnn_handle_;

    cuda_internal::CusolverDnHandle cusolverdn_handle_;

    cuda_internal::CudaConv cuda_conv_{};
};

DeviceInternals& GetDeviceInternals(CudaDevice& device);

}  // namespace cuda_internal

struct CudaBatchNormGradState : public BatchNormGradState {
public:
    CudaBatchNormGradState(Array x_cont, Array x_mean, Array x_inv_std, Dtype beta_dtype)
        : x_cont_{std::move(x_cont)}, x_mean_{std::move(x_mean)}, x_inv_std_{std::move(x_inv_std)}, beta_dtype_{beta_dtype} {}

    const Array& x_cont() const { return x_cont_; }
    const Array& x_mean() const { return x_mean_; }
    const Array& x_inv_std() const { return x_inv_std_; }
    Dtype beta_dtype() const { return beta_dtype_; }

private:
    Array x_cont_;
    Array x_mean_;
    Array x_inv_std_;
    Dtype beta_dtype_;
};

struct GenericRnnGradState : public RnnGradState {
    GenericRnnGradState(cudnnRNNDescriptor_t rnn_desc, cudnnFilterDescriptor_t w_desc, Array w, Array reserve, Array workspace)
        : rnn_desc_{rnn_desc}, w_desc_{w_desc}, w_{std::move(w)}, reserve_{std::move(reserve)}, workspace_{std::move(workspace)} {}
    cudnnRNNDescriptor_t rnn_desc() { return rnn_desc_; }
    cudnnFilterDescriptor_t wDesc() { return w_desc_; }
    Array w() { return w_; }
    Array reserve() { return reserve_; }
    Array workspace() { return workspace_; }

private:
    cudnnRNNDescriptor_t rnn_desc_;
    cudnnFilterDescriptor_t w_desc_;
    Array w_;
    Array reserve_;
    Array workspace_;
};

// Pooling states are identical for most CUDA pooling ops so we define a common base class.
class CudaPoolStateBase {
public:
    CudaPoolStateBase(Array x, Array out) : x_{std::move(x)}, out_{std::move(out)} {}

    const Array& x() const { return x_; }
    const Array& out() const { return out_; }

private:
    Array x_{};
    Array out_{};
};

class CudaMaxPoolGradState : public MaxPoolGradState, public CudaPoolStateBase {
    using CudaPoolStateBase::CudaPoolStateBase;
};

class CudaMaxPoolGradGradState : public MaxPoolGradGradState, public CudaPoolStateBase {
    using CudaPoolStateBase::CudaPoolStateBase;
};

class CudaAveragePoolGradState : public AveragePoolGradState, public CudaPoolStateBase {
    using CudaPoolStateBase::CudaPoolStateBase;
};

class CudaDevice : public Device {
public:
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

protected:
    CudaDevice(CudaBackend& backend, int index)
        : Device{backend, index},
          device_memory_pool_{std::make_shared<MemoryPool>(index, std::make_unique<DeviceMemoryAllocator>())},
          pinned_memory_pool_{std::make_shared<MemoryPool>(index, std::make_unique<PinnedMemoryAllocator>())},
          device_internals_{index} {}

private:
    friend CudaDevice* cuda_internal::CreateDevice(CudaBackend& backend, int index);

    friend cuda_internal::DeviceInternals& cuda_internal::GetDeviceInternals(CudaDevice& device);

    friend class cuda_internal::CudaConvTest;  // for unit-tests

    // Allocates pinned memory.
    // The pinned memory is used internally by the CUDA device for asynchronous memory transfer, i.e. cudaMemcpyAsync.
    std::shared_ptr<void> AllocatePinnedMemory(size_t bytesize);

    // Asynchronous transfer from host to this device, w.r.t. host, using temporary pinned memory.
    // The current device must be set to this device, prior to calling this function.
    void MemoryCopyFromHostAsync(void* dst, const void* src, size_t bytesize);

    std::shared_ptr<MemoryPool> device_memory_pool_;

    // TODO(hvy): Consider checking if pinned memory is available by querying canMapHostMemory.
    std::shared_ptr<MemoryPool> pinned_memory_pool_;

    cuda_internal::DeviceInternals device_internals_;

    // Memory keeper.
    cuda_internal::MemoryKeeper memory_keeper_{};
};

}  // namespace cuda
}  // namespace chainerx
