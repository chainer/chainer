#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include "xchainer/array_body.h"
//#include "xchainer/device.h"
#include "xchainer/dtype.h"
//#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {

// Handle of ArrayBody.
//
// Users basically use Array to use a multi-dimensional array.
class Array {
 public:
  Array() = default;
  explicit Array(std::shared_ptr<ArrayBody> body) : body_(std::move(body)) {}

  //
  // Query for ArrayBody
  //

  bool is_null() const {
    return body_ == nullptr;
  }

  ArrayBody* raw_body() {
    return body_.get();
  }

  const std::shared_ptr<ArrayBody>& body() const {
    return body_;
  }

  void set_body(std::shared_ptr<ArrayBody> body) {
    body_ = std::move(body);
  }

  std::shared_ptr<ArrayBody> move_body() {
    return std::move(body_);
  }

  //
  // Shortcut accessors of ArrayBody. They do not check if body_ is not null.
  //

  // Device device() const {
  //   return body_->device();
  // }

  Dtype dtype() const {
    return body_->dtype();
  }

  int8_t ndim() const {
    return body_->ndim();
  }

  const Shape& shape() const {
    return body_->shape();
  }

  int64_t total_size() const {
    return body_->total_size();
  }

  int64_t element_bytes() const {
    return body_->element_bytes();
  }

  int64_t total_bytes() const {
    return body_->total_bytes();
  }

  bool is_contiguous() const {
    return body_->is_contiguous();
  }

  //
  // Operators as member functions
  //

  //Array& operator+=(const Array& other);
  //Array& operator*=(const Array& other);

  //
  // Array manipulation as member functions
  //

  // void Fill(Scalar value);

 private:
  std::shared_ptr<ArrayBody> body_;
};

//
// Operators as free functions
//

//Array operator+(const Array& lhs, const Array& rhs);
//Array operator*(const Array& lhs, const Array& rhs);

}  // namespace xchainer
