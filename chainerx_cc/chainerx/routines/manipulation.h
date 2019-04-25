#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/shape.h"
#include "chainerx/scalar.h"

namespace chainerx {

// TODO(niboshi) : Implement Edge and Reflection as pad modes
enum class PadMode {
    kConstant = 1,
};

enum class PadDtypeKind {
  kSingle = 1,
  kVector,
  kVectorOfTuple,
};

class PadWidths{
private:
    int64_t int_pad_width_;
    std::vector<int64_t> vec_pad_width_;
    std::vector<std::vector<int64_t>> vec_tup_pad_width_;

  PadDtypeKind kind_;

public:
  PadWidths() = default;
  PadWidths(int64_t pw): int_pad_width_{pw}, kind_{PadDtypeKind::kSingle} {}  // NOLINT(runtime/explicit)
  PadWidths(const std::vector<int64_t>& pw): vec_pad_width_{pw}, kind_{PadDtypeKind::kVector} {}  // NOLINT(runtime/explicit)
  PadWidths(const std::vector<std::vector<int64_t>>& pw):  // NOLINT(runtime/explicit)
            vec_tup_pad_width_{pw},
            kind_{PadDtypeKind::kVectorOfTuple}
            {}

  // Convert single integer and vector of integer values to vector of two element vectors
  std::vector<std::vector<int64_t>> GetVectorOfTuple(const Array& array){
      std::vector<std::vector<int64_t>> new_pad_width(array.ndim());
      switch (kind_){
          case PadDtypeKind::kSingle:
              for (int8_t i = 0; i < array.ndim(); ++i){
                std::vector<int64_t> tuple = {int_pad_width_, int_pad_width_};
                new_pad_width[i] = tuple;
              }
              break;
          case PadDtypeKind::kVector:
              for (int8_t i = 0; i < array.ndim(); ++i){
                std::vector<int64_t> tuple = {vec_pad_width_[0], vec_pad_width_[1]};
                new_pad_width[i] = tuple;
              }
              break;
          case PadDtypeKind::kVectorOfTuple:
              new_pad_width = vec_tup_pad_width_;
              break;
          default:
              CHAINERX_NEVER_REACH();
      }
      return new_pad_width;
  }
};

class ConstantValues{
private:
    int64_t int_constant_values_;
    std::vector<Scalar> vec_constant_values_;
    std::vector<std::vector<Scalar>> vec_tup_constant_values_;

  PadDtypeKind kind_;

public:
  ConstantValues() = default;
  ConstantValues(Scalar cv): int_constant_values_{cv}, kind_{PadDtypeKind::kSingle} {}  // NOLINT(runtime/explicit)
  ConstantValues(const std::vector<Scalar>& cv): vec_constant_values_{cv}, kind_{PadDtypeKind::kVector} {}  // NOLINT(runtime/explicit)
  ConstantValues(const std::vector<std::vector<Scalar>>& cv):  // NOLINT(runtime/explicit)
                 vec_tup_constant_values_{cv},
                 kind_{PadDtypeKind::kVectorOfTuple}
                 {}

  // Convert single integer and vector of integer values to vector of two element vectors
  std::vector<std::vector<Scalar>> GetVectorOfTuple(const Array& array){
      std::vector<std::vector<Scalar>> new_constant_values(array.ndim());
      switch (kind_){
          case PadDtypeKind::kSingle:
              for (int8_t i = 0; i < array.ndim(); ++i){
                std::vector<Scalar> tuple = {int_constant_values_, int_constant_values_};
                new_constant_values[i] = tuple;
              }
              break;
          case PadDtypeKind::kVector:
              for (int8_t i = 0; i < array.ndim(); ++i){
                std::vector<Scalar> tuple = {vec_constant_values_[0], vec_constant_values_[1]};
                new_constant_values[i] = tuple;
              }
              break;
          case PadDtypeKind::kVectorOfTuple:
              new_constant_values = vec_tup_constant_values_;
              break;
          default:
              CHAINERX_NEVER_REACH();
      }
      return new_constant_values;
  }
};
// Retrieves a scalar from a single-element array.
//
// If the array is not single-element, DimensionError is thrown.
Scalar AsScalar(const Array& a);

// Returns a view where the specified axis is moved to start.
Array RollAxis(const Array& a, int8_t axis, int8_t start = 0);

// Returns a transposed view of the array.
Array Transpose(const Array& a, const OptionalAxes& axes = nonstd::nullopt);

// Returns an array padded by pad_width amount
Array Pad(const Array& a, PadWidths pad_width, PadMode mode, ConstantValues constant_values);
// Array Pad(const Array& a, std::vector<int64_t> pad_width, PadMode mode, std::vector<Scalar> constant_values);
// Array Pad(const Array& a, std::vector<std::vector<int64_t>> pad_width, PadMode mode, std::vector<std::vector<Scalar>> constant_values);

// Returns a reshaped array.
Array Reshape(const Array& a, const Shape& newshape);

// Returns a squeezed array with unit-length axes removed.
//
// If no axes are specified, all axes of unit-lengths are removed.
// If no axes can be removed, an array with aliased data is returned.
Array Squeeze(const Array& a, const OptionalAxes& axis = nonstd::nullopt);

// Broadcasts the array to the specified shape.
// Returned array is always a view to this array.
Array BroadcastTo(const Array& array, const Shape& shape);

// Returns a concatenated array.
Array Concatenate(const std::vector<Array>& arrays);
Array Concatenate(const std::vector<Array>& arrays, nonstd::optional<int8_t> axis);

// Returns a joined array along a new axis.
Array Stack(const std::vector<Array>& arrays, int8_t axis = 0);

// Returns a set of arrays resulting from splitting the given array into sections along the specified axis.
// If the dimension is not equally divisible, DimensionError is throws.
std::vector<Array> Split(const Array& ary, int64_t sections, int8_t axis = 0);

// Returns a set of arrays resulting from splitting the given array at the indices along the specified axis.
std::vector<Array> Split(const Array& ary, std::vector<int64_t> indices, int8_t axis = 0);

}  // namespace chainerx
