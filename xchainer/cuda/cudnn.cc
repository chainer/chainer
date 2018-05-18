#include "xchainer/cuda/cudnn.h"

#include <cudnn.h>

#include "xchainer/error.h"

namespace xchainer {
namespace cuda {

CudnnError::CudnnError(cudnnStatus_t status) : XchainerError{cudnnGetErrorString(status)}, status_{status} {}

void CheckCudnnError(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw CudnnError{status};
    }
}

}  // namespace cuda
}  // namespace xchainer
