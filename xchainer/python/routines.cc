#include "xchainer/python/routines.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/linalg.h"
#include "xchainer/routines/logic.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/routines/math.h"
#include "xchainer/routines/sorting.h"
#include "xchainer/scalar.h"

#include "xchainer/python/array.h"
#include "xchainer/python/array_index.h"
#include "xchainer/python/common.h"
#include "xchainer/python/device.h"
#include "xchainer/python/shape.h"
#include "xchainer/python/strides.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

namespace {

ArrayBodyPtr MakeArray(const py::list& list, const nonstd::optional<Dtype>& dtype, Device& device) {
    // TODO(sonots): Determine dtype (bool or int64, or float64) seeing values of list.
    // TODO(sonots): Support nested list
    py::tuple shape_tup{1};
    shape_tup[0] = list.size();
    return internal::MakeArray(shape_tup, dtype.value_or(Dtype::kFloat64), list, device);
}

}  // namespace
void InitXchainerRoutines(pybind11::module& m) {
    // creation routines
    m.def("array",
          [](const py::list& list, const nonstd::optional<Dtype>& dtype, const nonstd::optional<std::string>& device_id) {
              return MakeArray(list, dtype, GetDevice(device_id));
          },
          py::arg("object"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("array",
          [](const py::list& list, const nonstd::optional<Dtype>& dtype, Device& device) { return MakeArray(list, dtype, device); },
          py::arg("object"),
          py::arg("dtype") = nullptr,
          py::arg("device"));
    m.def("array",
          [](const py::array& array, const nonstd::optional<std::string>& device_id) { return MakeArray(array, GetDevice(device_id)); },
          py::arg("object"),
          py::arg("device") = nullptr);
    m.def("array", [](const py::array& array, Device& device) { return MakeArray(array, device); }, py::arg("object"), py::arg("device"));
    // Returns a view of an array if device argument is not specified.
    // Returns a new array transferred to the given device if device argument is specified. Note that the graph is connected.
    m.def("array",
          [](const ArrayBodyPtr& array, const nonstd::optional<std::string>& device_id) {
              if (device_id) {
                  return Array{array}.ToDevice(GetDevice(device_id)).move_body();
              }
              return Array{array}.MakeView().move_body();
          },
          py::arg("object"),
          py::arg("device") = nullptr);
    m.def("array",
          [](const ArrayBodyPtr& array, Device& device) { return Array{array}.ToDevice(device).move_body(); },
          py::arg("object"),
          py::arg("device"));
    m.def("empty",
          [](py::tuple shape, Dtype dtype, const nonstd::optional<std::string>& device_id) {
              return Array::Empty(ToShape(shape), dtype, GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("empty",
          [](py::tuple shape, Dtype dtype, Device& device) { return Array::Empty(ToShape(shape), dtype, device).move_body(); },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"));
    m.def("full",
          [](py::tuple shape, Scalar fill_value, Dtype dtype, const nonstd::optional<std::string>& device_id) {
              return Array::Full(ToShape(shape), fill_value, dtype, GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("full",
          [](py::tuple shape, Scalar fill_value, Dtype dtype, Device& device) {
              return Array::Full(ToShape(shape), fill_value, dtype, device).move_body();
          },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("dtype"),
          py::arg("device"));
    m.def("full",
          [](py::tuple shape, Scalar fill_value, const nonstd::optional<std::string>& device_id) {
              return Array::Full(ToShape(shape), fill_value, GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("device") = nullptr);
    m.def("full",
          [](py::tuple shape, Scalar fill_value, Device& device) { return Array::Full(ToShape(shape), fill_value, device).move_body(); },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("device"));
    m.def("zeros",
          [](py::tuple shape, Dtype dtype, const nonstd::optional<std::string>& device_id) {
              return Array::Zeros(ToShape(shape), dtype, GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("zeros",
          [](py::tuple shape, Dtype dtype, Device& device) { return Array::Zeros(ToShape(shape), dtype, device).move_body(); },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"));
    m.def("ones",
          [](py::tuple shape, Dtype dtype, const nonstd::optional<std::string>& device_id) {
              return Array::Ones(ToShape(shape), dtype, GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("ones",
          [](py::tuple shape, Dtype dtype, Device& device) { return Array::Ones(ToShape(shape), dtype, device).move_body(); },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"));
    m.def("empty_like",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::string>& device_id) {
              return Array::EmptyLike(Array{a}, GetDevice(device_id)).move_body();
          },
          py::arg("a"),
          py::arg("device") = nullptr);
    m.def("empty_like",
          [](const ArrayBodyPtr& a, Device& device) { return Array::EmptyLike(Array{a}, device).move_body(); },
          py::arg("a"),
          py::arg("device"));
    m.def("full_like",
          [](const ArrayBodyPtr& a, Scalar value, const nonstd::optional<std::string>& device_id) {
              return Array::FullLike(Array{a}, value, GetDevice(device_id)).move_body();
          },
          py::arg("a"),
          py::arg("fill_value"),
          py::arg("device") = nullptr);
    m.def("full_like",
          [](const ArrayBodyPtr& a, Scalar value, Device& device) { return Array::FullLike(Array{a}, value, device).move_body(); },
          py::arg("a"),
          py::arg("fill_value"),
          py::arg("device"));
    m.def("zeros_like",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::string>& device_id) {
              return Array::ZerosLike(Array{a}, GetDevice(device_id)).move_body();
          },
          py::arg("a"),
          py::arg("device") = nullptr);
    m.def("zeros_like",
          [](const ArrayBodyPtr& a, Device& device) { return Array::ZerosLike(Array{a}, device).move_body(); },
          py::arg("a"),
          py::arg("device"));
    m.def("ones_like",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::string>& device_id) {
              return Array::OnesLike(Array{a}, GetDevice(device_id)).move_body();
          },
          py::arg("a"),
          py::arg("device") = nullptr);
    m.def("ones_like",
          [](const ArrayBodyPtr& a, Device& device) { return Array::OnesLike(Array{a}, device).move_body(); },
          py::arg("a"),
          py::arg("device"));
    m.def("copy", [](const ArrayBodyPtr& a) { return Copy(Array{a}).move_body(); }, py::arg("a"));

    // linalg routines
    m.def("dot",
          [](const ArrayBodyPtr& a, const ArrayBodyPtr& b) { return Dot(Array{a}, Array{b}).move_body(); },
          py::arg("a"),
          py::arg("b"));

    // logic routines
    m.def("equal",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return Equal(Array{x1}, Array{x2}).move_body(); },
          py::arg("x1"),
          py::arg("x2"));

    // manipulation routines
    m.def("asscalar",
          [](const ArrayBodyPtr& a) -> py::object {
              Scalar s = AsScalar(Array{a});
              switch (GetKind(s.dtype())) {
                  case DtypeKind::kBool:
                      return py::bool_{static_cast<bool>(s)};
                  case DtypeKind::kInt:
                      // fallthrough
                  case DtypeKind::kUInt:
                      return py::int_{static_cast<int64_t>(s)};
                  case DtypeKind::kFloat:
                      return py::float_{static_cast<double>(s)};
                  default:
                      assert(false);  // never reach
              }
          },
          py::arg("a"));
    m.def("transpose", [](const ArrayBodyPtr& a) { return Transpose(Array{a}).move_body(); }, py::arg("a"));
    m.def("reshape",
          [](const ArrayBodyPtr& a, py::tuple newshape) { return Reshape(Array{a}, ToShape(newshape)).move_body(); },
          py::arg("a"),
          py::arg("newshape"));
    m.def("reshape",
          [](const ArrayBodyPtr& a, const std::vector<int64_t>& newshape) {
              return Reshape(Array{a}, {newshape.begin(), newshape.end()}).move_body();
          },
          py::arg("a"),
          py::arg("newshape"));
    m.def("squeeze",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::vector<int8_t>>& axis) { return Squeeze(Array{a}, axis).move_body(); },
          py::arg("a"),
          py::arg("axis") = nullptr);
    m.def("squeeze",
          [](const ArrayBodyPtr& a, int8_t axis) { return Squeeze(Array{a}, std::vector<int8_t>{axis}).move_body(); },
          py::arg("a"),
          py::arg("axis"));
    m.def("broadcast_to",
          [](const ArrayBodyPtr& array, py::tuple shape) { return Array{array}.BroadcastTo(ToShape(shape)).move_body(); },
          py::arg("array"),
          py::arg("shape"));
    m.def("broadcast_to",
          [](const ArrayBodyPtr& array, py::tuple shape) { return Array{array}.BroadcastTo(ToShape(shape)).move_body(); },
          py::arg("array"),
          py::arg("shape"));

    // math routines
    m.def("add",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return (Array{x1} + Array{x2}).move_body(); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("subtract",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return (Array{x1} - Array{x2}).move_body(); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("multiply",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return (Array{x1} * Array{x2}).move_body(); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("multiply", [](const ArrayBodyPtr& x1, Scalar x2) { return Multiply(Array{x1}, x2).move_body(); }, py::arg("x1"), py::arg("x2"));
    m.def("multiply", [](Scalar x1, const ArrayBodyPtr& x2) { return Multiply(x1, Array{x2}).move_body(); }, py::arg("x1"), py::arg("x2"));
    m.def("divide",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return (Array{x1} / Array{x2}).move_body(); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("sum",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return Sum(Array{a}, std::vector<int8_t>{axis}, keepdims).move_body(); },
          py::arg("a"),
          py::arg("axis"),
          py::arg("keepdims") = false);
    m.def("sum",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return Sum(Array{a}, axis, keepdims).move_body();
          },
          py::arg("a"),
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    m.def("maximum", [](const ArrayBodyPtr& x1, Scalar x2) { return Maximum(Array{x1}, x2).move_body(); }, py::arg("x1"), py::arg("x2"));
    m.def("maximum", [](Scalar x1, const ArrayBodyPtr& x2) { return Maximum(x1, Array{x2}).move_body(); }, py::arg("x1"), py::arg("x2"));

    // sorting routines
    m.def("argmax",
          [](const ArrayBodyPtr& a, const nonstd::optional<int8_t>& axis) { return ArgMax(Array{a}, axis).move_body(); },
          py::arg("a"),
          py::arg("axis") = nullptr);
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
