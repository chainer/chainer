#include "xchainer/cuda/array_math.h"

#include <cassert>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"

namespace xchainer {
namespace cuda {

namespace {

template <typename T>
__global__ void AddKernel(const T* ldata, const T* rdata, T* odata, const int64_t total_size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        odata[i] = ldata[i] + rdata[i];
    }
}

template <typename T>
__global__ void MulKernel(const T* ldata, const T* rdata, T* odata, const int64_t total_size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        odata[i] = ldata[i] * rdata[i];
    }
}

// TODO(sonots): support stream
template <typename T>
void Add(const Array& lhs, const Array& rhs, Array& out) {
    static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&AddKernel<T>).block_size;

    const int64_t total_size = lhs.shape().total_size();
    const int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
    const int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

    const T* ldata = static_cast<const T*>(lhs.data().get());
    const T* rdata = static_cast<const T*>(rhs.data().get());
    T* odata = static_cast<T*>(out.data().get());
    AddKernel<<<grid_size, block_size>>>(ldata, rdata, odata, total_size);
}

// TODO(sonots): support stream
template <typename T>
void Mul(const Array& lhs, const Array& rhs, Array& out) {
    static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&MulKernel<T>).block_size;

    const int64_t total_size = lhs.shape().total_size();
    const int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
    const int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

    const T* ldata = static_cast<const T*>(lhs.data().get());
    const T* rdata = static_cast<const T*>(rhs.data().get());
    T* odata = static_cast<T*>(out.data().get());
    MulKernel<<<grid_size, block_size>>>(ldata, rdata, odata, total_size);
}

}  // namespace

void Add(const Array& lhs, const Array& rhs, Array& out) {
    switch (lhs.dtype()) {
        case Dtype::kBool:
            Add<bool>(lhs, rhs, out);
            break;
        case Dtype::kInt8:
            Add<int8_t>(lhs, rhs, out);
            break;
        case Dtype::kInt16:
            Add<int16_t>(lhs, rhs, out);
            break;
        case Dtype::kInt32:
            Add<int32_t>(lhs, rhs, out);
            break;
        case Dtype::kInt64:
            Add<int64_t>(lhs, rhs, out);
            break;
        case Dtype::kUInt8:
            Add<uint8_t>(lhs, rhs, out);
            break;
        case Dtype::kFloat32:
            Add<float>(lhs, rhs, out);
            break;
        case Dtype::kFloat64:
            Add<double>(lhs, rhs, out);
            break;
        default:
            assert(0);  // should never be reached
    }
}

void Mul(const Array& lhs, const Array& rhs, Array& out) {
    switch (lhs.dtype()) {
        case Dtype::kBool:
            Mul<bool>(lhs, rhs, out);
            break;
        case Dtype::kInt8:
            Mul<int8_t>(lhs, rhs, out);
            break;
        case Dtype::kInt16:
            Mul<int16_t>(lhs, rhs, out);
            break;
        case Dtype::kInt32:
            Mul<int32_t>(lhs, rhs, out);
            break;
        case Dtype::kInt64:
            Mul<int64_t>(lhs, rhs, out);
            break;
        case Dtype::kUInt8:
            Mul<uint8_t>(lhs, rhs, out);
            break;
        case Dtype::kFloat32:
            Mul<float>(lhs, rhs, out);
            break;
        case Dtype::kFloat64:
            Mul<double>(lhs, rhs, out);
            break;
        default:
            assert(0);  // should never be reached
    }
}

}  // namespace cuda
}  // namespace xchainer
