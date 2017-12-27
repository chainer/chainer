#include "xchainer/cuda/array_math.h"

#include <cassert>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"

namespace xchainer {
namespace cuda {

namespace {

template <typename T>
__global__ void FillKernel(T* odata, const T value, const int64_t total_size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        odata[i] = value;
    }
}

template <typename T>
void FillImpl(Array out, T value) {
    static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&FillKernel<T>).block_size;

    const int64_t total_size = out.shape().total_size();
    const int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
    const int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

    T* odata = static_cast<T*>(out.data().get());
    FillKernel<<<grid_size, block_size>>>(odata, value, total_size);
}

}  // namespace

void Fill(Array& out, Scalar value) {
    switch (value.dtype()) {
        case Dtype::kBool:
            FillImpl(out, static_cast<bool>(value));
            break;
        case Dtype::kInt8:
            FillImpl(out, static_cast<int8_t>(value));
            break;
        case Dtype::kInt16:
            FillImpl(out, static_cast<int16_t>(value));
            break;
        case Dtype::kInt32:
            FillImpl(out, static_cast<int32_t>(value));
            break;
        case Dtype::kInt64:
            FillImpl(out, static_cast<int64_t>(value));
            break;
        case Dtype::kUInt8:
            FillImpl(out, static_cast<uint8_t>(value));
            break;
        case Dtype::kFloat32:
            FillImpl(out, static_cast<float>(value));
            break;
        case Dtype::kFloat64:
            FillImpl(out, static_cast<double>(value));
            break;
        default:
            assert(false);  // should never be reached
    }
}

}  // namespace cuda
}  // namespace xchainer
