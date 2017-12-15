#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <gsl/gsl>
//#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/shape.h"

namespace xchainer {

// The main data structure of multi-dimensional array.
class ArrayBody {
 public:
  //ArrayBody(Device device, gsl::span<const int64_t> shape, Dtype dtype);

  //ArrayBody(Device device, const Shape& shape, Dtype dtype)
  //    : ArrayBody(device, shape.span(), dtype) {}

  //Device device() const {
  //  return device_;
  //}

  ArrayBody(gsl::span<const int64_t> shape, Dtype dtype);

  ArrayBody(const Shape& shape, Dtype dtype)
      : ArrayBody(shape.span(), dtype) {}

  Dtype dtype() const {
    return dtype_;
  }

  int8_t ndim() const {
    return shape_.ndim();
  }

  const Shape& shape() const {
    return shape_;
  }

  //const Strides& strides() const {
  //  return strides_;
  //}

  bool is_contiguous() const {
    return is_contiguous_;
  }

  int64_t total_size() const {
    return shape_.total_size();
  }

  int64_t element_bytes() const {
    return GetElementSize(dtype_);
  }

  int64_t total_bytes() const {
    return total_size() * element_bytes();
  }

  const std::shared_ptr<void>& data() const {
    return data_;
  }

  void* raw_data() const {
    return data_.get();
  }

  int64_t offset() const {
    return offset_;
  }

  //void SetData(std::shared_ptr<void> data, gsl::span<const int64_t> strides, int64_t offset = 0);

  //void SetData(ArrayBody& other, gsl::span<const int64_t> strides, int64_t relative_offset = 0) {
  //  SetData(other.data_, strides, other.offset_ + relative_offset);
  //}

  void SetContiguousData(std::shared_ptr<void> data, int64_t offset = 0);

  void SetContiguousData(ArrayBody& other, int64_t relative_offset = 0) {
    SetContiguousData(other.data_, other.offset_ + relative_offset);
  }

  std::shared_ptr<ArrayBody> MakeSimilar() const {
    //return std::make_shared<ArrayBody>(device_, shape_, dtype_);
    return std::make_shared<ArrayBody>(shape_, dtype_);
  }

 private:
  //Device device_;

  Shape shape_;
  //Strides strides_;
  bool is_contiguous_;

  Dtype dtype_;

  std::shared_ptr<void> data_;
  int64_t offset_;
};

}  // namespace xchainer
