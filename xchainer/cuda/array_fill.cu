#include "xchainer/cuda/array_math.h"

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/dtype.h"

namespace xchainer {
namespace cuda {

namespace {

template <typename T>
__global__ void FillKernel(T* odata, T value, int64_t total_size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total_size; i += blockDim.x * gridDim.x) {
        odata[i] = value;
    }
}

}  // namespace

void Fill(Array& out, Scalar value) {
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&FillKernel<T>).block_size;

        int64_t total_size = out.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        auto* odata = static_cast<T*>(out.data().get());
        FillKernel<<<grid_size, block_size>>>(odata, static_cast<T>(value), total_size);
    });
}

}  // namespace cuda
}  // namespace xchainer
