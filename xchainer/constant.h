#pragma once

namespace xchainer {

// Special ndim value that indicates ndim is determined at runtime. This value is used as a template parameter of, for example,
// IndexableArray and Indexer.
constexpr int8_t kDynamicNdim = -1;

// Maximum number of dimensions (axes) of each array.
constexpr int8_t kMaxNdim = 8;

// Default graph ID to be used if not specified
constexpr const char* kDefaultGraphId = "default";

}  // namespace xchainer
