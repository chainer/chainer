#pragma once

#include <mutex>

namespace xchainer {
namespace cuda {
namespace cuda_internal {

extern std::mutex* g_mutex;

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace xchainer
