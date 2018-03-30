#include "xchainer/cuda/cuda_device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/native_device.h"
#include "xchainer/scalar.h"

namespace xchainer {
namespace cuda {

namespace {

static constexpr int kMaxReductionBlockSize = 512;

int64_t RoundUpToPowerOf2(int64_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}
}  // namespace

namespace {

template <typename T>
__global__ void FillKernel(IndexableArray<T> out_iarray, T value, Indexer<> indexer) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size(); i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        out_iarray[indexer] = value;
    }
}

template <typename T>
__global__ void SumKernel(
        IndexableArray<const T> src_iarray,
        IndexableArray<T> out_iarray,
        Indexer<> src_indexer,
        Indexer<> reduce_indexer,
        Indexer<> out_indexer,
        int reduce_block_size) {
    extern __shared__ __align__(8) uint8_t work_bytes[];
    T* work = reinterpret_cast<T*>(work_bytes);
    int tid = threadIdx.x;
    int reduce_blocks_per_grid = (blockDim.x + reduce_block_size - 1) / reduce_block_size * gridDim.x;

    for (int64_t i_out = blockIdx.x; i_out < out_indexer.total_size(); i_out += gridDim.x * reduce_blocks_per_grid) {
        out_indexer.Set(i_out);

        T sum_value = 0;

        // Set output indices in the corresponding indices (out_axis) in src_index.
        for (int8_t i_out_dim = 0; i_out_dim < out_indexer.ndim(); ++i_out_dim) {
            src_indexer.index()[i_out_dim] = out_indexer.index()[i_out_dim];
        }

        // Linearly compute the partial sum into at most kMaxReductionBlockSize values.
        for (int64_t i_reduce = tid; i_reduce < reduce_indexer.total_size(); i_reduce += reduce_block_size) {
            reduce_indexer.Set(i_reduce);

            // Set reduction indices in the corresponding indices (axis) in src_index.
            for (int8_t i_reduce_dim = 0; i_reduce_dim < reduce_indexer.ndim(); ++i_reduce_dim) {
                src_indexer.index()[out_indexer.ndim() + i_reduce_dim] = reduce_indexer.index()[i_reduce_dim];
            }

            sum_value += src_iarray[src_indexer];
        }

        if (reduce_block_size >= 2) {
            // Synchronize partial sums
            work[tid] = sum_value;
            __syncthreads();

            // Reduction
            if (reduce_block_size > 2) {
                if (reduce_block_size > 4) {
                    if (reduce_block_size > 8) {
                        if (reduce_block_size > 16) {
                            if (reduce_block_size > 32) {
                                if (reduce_block_size > 64) {
                                    if (reduce_block_size > 128) {
                                        if (reduce_block_size > 256) {
                                            static_assert(kMaxReductionBlockSize == 512, "");

                                            if (tid < 256) {
                                                work[tid] += work[tid + 256];
                                            }
                                            __syncthreads();
                                        }
                                        if (tid < 128) {
                                            work[tid] += work[tid + 128];
                                        }
                                        __syncthreads();
                                    }
                                    if (tid < 64) {
                                        work[tid] += work[tid + 64];
                                    }
                                    __syncthreads();
                                }
                                if (tid < 32) {
                                    work[tid] += work[tid + 32];
                                }
                                __syncthreads();
                            }
                            if (tid < 16) {
                                work[tid] += work[tid + 16];
                            }
                            __syncthreads();
                        }
                        if (tid < 8) {
                            work[tid] += work[tid + 8];
                        }
                        __syncthreads();
                    }
                    if (tid < 4) {
                        work[tid] += work[tid + 4];
                    }
                    __syncthreads();
                }
                if (tid < 2) {
                    work[tid] += work[tid + 2];
                }
                __syncthreads();
            }
            sum_value = work[0] + work[1];
        }
        // Store the output value
        if (tid == 0) {
            out_iarray[out_indexer] = sum_value;
        }
    }
}

template <typename T>
__global__ void CopyKernel(IndexableArray<const T> src_iarray, IndexableArray<T> out_iarray, Indexer<> indexer) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size(); i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        out_iarray[indexer] = src_iarray[indexer];
    }
}

template <typename T>
__global__ void EqualKernel(
        IndexableArray<const T> lhs_iarray, IndexableArray<const T> rhs_iarray, IndexableArray<bool> out_iarray, Indexer<> indexer) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        out_iarray[indexer] = lhs_iarray[indexer] == rhs_iarray[indexer];
    }
}

template <typename T>
__global__ void AddKernel(
        IndexableArray<const T> lhs_iarray, IndexableArray<const T> rhs_iarray, IndexableArray<T> out_iarray, Indexer<> indexer) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        out_iarray[indexer] = lhs_iarray[indexer] + rhs_iarray[indexer];
    }
}

template <typename T>
__global__ void MulScalarKernel(IndexableArray<const T> lhs_iarray, T rhs_value, IndexableArray<T> out_iarray, Indexer<> indexer) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        out_iarray[indexer] = lhs_iarray[indexer] * rhs_value;
    }
}

template <typename T>
__global__ void MulKernel(
        IndexableArray<const T> lhs_iarray, IndexableArray<const T> rhs_iarray, IndexableArray<T> out_iarray, Indexer<> indexer) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        out_iarray[indexer] = lhs_iarray[indexer] * rhs_iarray[indexer];
    }
}

template <typename T>
__global__ void IfLessElseKernel(
        IndexableArray<const T> lhs_iarray,
        T rhs_value,
        T pos_value,
        IndexableArray<const T> neg_iarray,
        IndexableArray<T> out_iarray,
        Indexer<> indexer) {
    const int64_t total_size = indexer.total_size();
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        indexer.Set(i);
        out_iarray[indexer] = lhs_iarray[indexer] < rhs_value ? pos_value : neg_iarray[indexer];
    }
}

}  // namespace

std::shared_ptr<void> CudaDevice::Allocate(size_t bytesize) {
    if (bytesize == 0) {
        return nullptr;
    }
    CheckError(cudaSetDevice(index()));
    void* raw_ptr = nullptr;
    // Be careful to be exception-safe, i.e.,
    // do not throw any exceptions before creating shared_ptr when memory allocation is succeeded
    cudaError_t status = cudaMallocManaged(&raw_ptr, bytesize, cudaMemAttachGlobal);
    if (status != cudaSuccess) {
        cuda::Throw(status);
    }
    return std::shared_ptr<void>{raw_ptr, cudaFree};
}

void CudaDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    assert(IsPointerCudaMemory(dst));
    if (&src_device == this || nullptr != dynamic_cast<CudaDevice*>(&src_device)) {
        // Copy between CUDA devices
        CheckError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        assert(nullptr != dynamic_cast<native::NativeDevice*>(&src_device) &&
               "CudaDevice only supports copy between cuda or native devices.");
        // Copy from native device
        CheckError(cudaMemcpy(dst, src, bytesize, cudaMemcpyHostToDevice));
    }
}

void CudaDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    assert(IsPointerCudaMemory(src));
    if (&dst_device == this || nullptr != dynamic_cast<CudaDevice*>(&dst_device)) {
        // Copy between CUDA devices
        CheckError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        assert(nullptr != dynamic_cast<native::NativeDevice*>(&dst_device) &&
               "CudaDevice only supports copy between cuda or native devices.");
        // Copy to native device
        CheckError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToHost));
    }
}

std::shared_ptr<void> CudaDevice::TransferDataFrom(
        Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFrom(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, src_device);
    return dst_ptr;
}

std::shared_ptr<void> CudaDevice::TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = dst_device.Allocate(bytesize);
    MemoryCopyTo(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, dst_device);
    return dst_ptr;
}

std::shared_ptr<void> CudaDevice::FromContiguousData(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    CheckError(cudaMemcpy(dst_ptr.get(), src_ptr.get(), bytesize, cudaMemcpyHostToDevice));
    return dst_ptr;
}

void CudaDevice::Fill(const Array& out, Scalar value) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&FillKernel<T>).block_size;

        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{out.shape()};
        int64_t grid_size = (indexer.total_size() + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(indexer.total_size(), kMaxBlockSize);

        FillKernel<<<grid_size, block_size>>>(out_iarray, static_cast<T>(value), indexer);
    });
}

void CudaDevice::Sum(const Array& src, const std::vector<int8_t>& axis, const Array& out) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&SumKernel<T>).block_size;

        // Prepare indexable arrays and indexers
        auto tup = native::internal::PrepareIndexableArraysForReduction<T>(src, axis, out);
        IndexableArray<const T>& src_iarray = std::get<0>(tup);
        IndexableArray<T>& out_iarray = std::get<1>(tup);
        Indexer<>& src_indexer = std::get<2>(tup);
        Indexer<>& out_indexer = std::get<3>(tup);
        Indexer<>& reduce_indexer = std::get<4>(tup);

        // Launch kernel
        int reduce_block_size =
                static_cast<int>(std::min(static_cast<int64_t>(kMaxReductionBlockSize), RoundUpToPowerOf2(reduce_indexer.total_size())));
        int block_size = std::min(kMaxBlockSize, reduce_block_size);
        int64_t total_reduce_blocks = out_indexer.total_size();
        int64_t grid_size = total_reduce_blocks;
        size_t shared_mem_size = sizeof(T) * reduce_block_size;

        SumKernel<<<grid_size, block_size, shared_mem_size>>>(
                src_iarray, out_iarray, src_indexer, reduce_indexer, out_indexer, reduce_block_size);
    });
}

void CudaDevice::Copy(const Array& src, const Array& out) {
    CheckDevicesCompatible(src, out);
    cudaSetDevice(index());
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&CopyKernel<T>).block_size;

        IndexableArray<const T> src_iarray{src};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{out.shape()};

        int64_t total_size = indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        CopyKernel<<<grid_size, block_size>>>(src_iarray, out_iarray, indexer);
    });
}

void CudaDevice::Equal(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    cudaSetDevice(index());
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&EqualKernel<T>).block_size;

        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<bool> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        int64_t total_size = indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        EqualKernel<<<grid_size, block_size>>>(lhs_iarray, rhs_iarray, out_iarray, indexer);
    });
}

// TODO(sonots): support stream
void CudaDevice::Add(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    cudaSetDevice(index());
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&AddKernel<T>).block_size;

        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        int64_t total_size = indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        AddKernel<<<grid_size, block_size>>>(lhs_iarray, rhs_iarray, out_iarray, indexer);
    });
}

void CudaDevice::Mul(const Array& lhs, Scalar rhs, const Array& out) {
    CheckDevicesCompatible(lhs, out);
    cudaSetDevice(index());
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&MulScalarKernel<T>).block_size;

        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        int64_t total_size = indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        MulScalarKernel<<<grid_size, block_size>>>(lhs_iarray, static_cast<T>(rhs), out_iarray, indexer);
    });
}

// TODO(sonots): support stream
void CudaDevice::Mul(const Array& lhs, const Array& rhs, const Array& out) {
    CheckDevicesCompatible(lhs, rhs, out);
    cudaSetDevice(index());
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&MulKernel<T>).block_size;

        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        int64_t total_size = indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        MulKernel<<<grid_size, block_size>>>(lhs_iarray, rhs_iarray, out_iarray, indexer);
    });
}

void CudaDevice::IfLessElse(const Array& lhs, Scalar rhs, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(lhs, neg, out);
    cudaSetDevice(index());
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&IfLessElseKernel<T>).block_size;

        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> neg_iarray{neg};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};
        T rhs_value{rhs};
        T pos_value{pos};

        int64_t total_size = indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        IfLessElseKernel<<<grid_size, block_size>>>(lhs_iarray, rhs_value, pos_value, neg_iarray, out_iarray, indexer);
    });
}

void CudaDevice::Dot(const Array& lhs, const Array& rhs, const Array& out) {
    (void)lhs;  // unused
    (void)rhs;  // unused
    (void)out;  // unused
    throw NotImplementedError("CudaDevice::Dot is not yet implemented.");
}

void CudaDevice::Synchronize() {
    CheckError(cudaSetDevice(index()));
    CheckError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace xchainer
