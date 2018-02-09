#include "xchainer/cuda/array_math.h"

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/dtype.h"

namespace xchainer {
namespace cuda {

namespace {

template <typename T>
__global__ void AddKernel(const T* ldata, const T* rdata, T* odata, int64_t total_size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        odata[i] = ldata[i] + rdata[i];
    }
}

template <typename T>
__global__ void MulKernel(const T* ldata, const T* rdata, T* odata, int64_t total_size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        odata[i] = ldata[i] * rdata[i];
    }
}

}  // namespace

// TODO(sonots): support stream
void Add(const Array& lhs, const Array& rhs, Array& out) {
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&AddKernel<T>).block_size;

        int64_t total_size = lhs.TotalSize();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());
        AddKernel<<<grid_size, block_size>>>(ldata, rdata, odata, total_size);
    });
}

// TODO(sonots): support stream
void Mul(const Array& lhs, const Array& rhs, Array& out) {
    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&MulKernel<T>).block_size;

        int64_t total_size = lhs.TotalSize();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        auto* ldata = static_cast<const T*>(lhs.data().get());
        auto* rdata = static_cast<const T*>(rhs.data().get());
        auto* odata = static_cast<T*>(out.data().get());
        MulKernel<<<grid_size, block_size>>>(ldata, rdata, odata, total_size);
    });
}

}  // namespace cuda
}  // namespace xchainer
