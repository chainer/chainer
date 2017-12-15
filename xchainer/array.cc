#include "xchainer/array.h"

namespace xchainer {

Array::Array(gsl::span<const int64_t> shape, Dtype dtype) : shape_(shape), is_contiguous_(false), dtype_(dtype) {}

void Array::SetContiguousData(std::shared_ptr<void> data, int64_t offset) {
    data_ = std::move(data);
    offset_ = offset;
    is_contiguous_ = true;
}

}  // namespace xchainer
