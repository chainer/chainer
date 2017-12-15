#include "xchainer/array.h"

namespace xchainer {

//ArrayBody::ArrayBody(Device device, gsl::span<const int64_t> shape, Dtype dtype)
//    : device_(device),
//      shape_(shape),
//      is_contiguous_(false),
//      dtype_(dtype) {}

ArrayBody::ArrayBody(gsl::span<const int64_t> shape, Dtype dtype)
    : shape_(shape),
      is_contiguous_(false),
      dtype_(dtype) {}

//void ArrayBody::SetData(std::shared_ptr<void> data, gsl::span<const int64_t> strides, int64_t offset) {
//  const bool is_contiguous = IsContiguousLayout(shape_.cspan(), strides, element_bytes());
//  strides_.Reset(strides);
//  is_contiguous_ = is_contiguous;
//  data_ = std::move(data);
//  offset_ = offset;
//}

void ArrayBody::SetContiguousData(std::shared_ptr<void> data, int64_t offset) {
  //strides_.SetContiguous(shape_.cspan(), element_bytes());
  data_ = std::move(data);
  offset_ = offset;
  is_contiguous_ = true;
}

}  // namespace xchainer
