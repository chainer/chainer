#pragma once

#include <cstddef>

namespace chainerx {
namespace internal {

// Borrowed from boost::hash_combine
//
// See LICENSE.txt of ChainerX.
//
// TODO(sonots): hash combine in 64bit
inline void HashCombine(std::size_t& seed, std::size_t hash_value) { seed ^= hash_value + 0x9e3779b9 + (seed << 6) + (seed >> 2); }

}  // namespace internal
}  // namespace chainerx
