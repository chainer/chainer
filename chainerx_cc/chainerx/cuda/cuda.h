#pragma once

#include <mutex>

namespace chainerx {
namespace cuda {
namespace cuda_internal {

// TODO(imanishi): This is a temporary workaround for thread sanitizer causing data race errors for block sizes stored as static variables.
// Those variables will be moved to CudaDevice, and similarly the g_mutex.
// Delete this file and cuda.cc after that.

extern std::mutex* g_mutex;

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
