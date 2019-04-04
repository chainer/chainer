#include "chainerx/cuda/cuda.h"

#include <mutex>
#include <type_traits>

#include <gsl/gsl>

namespace chainerx {
namespace cuda {
namespace cuda_internal {

gsl::owner<std::mutex*> g_mutex = new std::mutex{};  // NOLINT(cert-err58-cpp)
static_assert(std::is_pod<decltype(g_mutex)>::value, "");

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace chainerx
