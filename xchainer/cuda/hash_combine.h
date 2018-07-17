#pragma once

#include <cstddef>

namespace xchainer {
namespace cuda {
namespace cuda_internal {

// Borrowed from boost::hash_combine
//
// See LICENSE.txt of xChainer.
//
// TODO(sonots): hash combine in 64bit
inline void HashCombine(std::size_t& seed, std::size_t hash_value) { seed ^= hash_value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace xchainer
