#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_runtime.h"

namespace xchainer {
namespace cuda {

void CudaBackend::Synchronize() { CheckError(cudaDeviceSynchronize()); }

}  // namespace cuda
}  // namespace xchainer
