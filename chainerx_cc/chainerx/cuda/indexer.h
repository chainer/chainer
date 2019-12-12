#pragma once

#include "chainerx/cuda/index_iterator.cuh"
#include "chainerx/indexer.h"


namespace chainerx {

template<int8_t kNdim = kDynamicNdim>
using CudaIndexer = Indexer<kNdim, CudaIndexIterator<kNdim>>;

}  // namespace chainerx
