#include "xchainer/cuda/cuda.h"

#include <mutex>
#include <type_traits>

namespace xchainer {
namespace cuda {
namespace cuda_internal {

std::mutex* g_mutex = new std::mutex{};
static_assert(std::is_pod<decltype(g_mutex)>::value, "");

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace xchainer
