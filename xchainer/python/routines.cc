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
#include "xchainer/ndim_vector.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/indexing.h"
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
#include "xchainer/python/dtype.h"
#include "xchainer/python/shape.h"
#include "xchainer/python/strides.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

namespace {

ArrayBodyPtr MakeArray(const py::list& list, py::handle dtype, Device& device) {
    // TODO(sonots): Determine dtype (bool or int64, or float64) seeing values of list.
    // TODO(sonots): Support nested list
    py::tuple shape_tup{1};
    shape_tup[0] = list.size();
    return internal::MakeArray(shape_tup, dtype.is_none() ? Dtype::kFloat64 : internal::GetDtype(dtype), list, device);
}

ArrayBodyPtr MakeArangeArray(
        Scalar start_or_stop,
        const nonstd::optional<Scalar>& maybe_stop,
        const nonstd::optional<Scalar>& maybe_step,
        py::handle dtype,
        Device& device) {
    Dtype start_or_stop_dtype = start_or_stop.dtype();
    Scalar start{0, start_or_stop_dtype};
    Scalar stop{start_or_stop};
    Scalar step = maybe_step.has_value() ? maybe_step.value() : Scalar{1, start_or_stop_dtype};

    if (maybe_stop.has_value()) {
        start = start_or_stop;
        stop = maybe_stop.value();
    }

    return dtype.is_none() ? Arange(start, stop, step, device).move_body()
                           : Arange(start, stop, step, internal::GetDtype(dtype), device).move_body();
}

}  // namespace

void InitXchainerRoutines(pybind11::module& m) {
    // creation routines
    m.def("array",
          [](const py::list& list, py::handle dtype, const nonstd::optional<std::string>& device_id) {
              return MakeArray(list, dtype, GetDevice(device_id));
          },
          py::arg("object"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("array",
          [](const py::list& list, py::handle dtype, Device& device) { return MakeArray(list, dtype, device); },
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
          [](py::tuple shape, py::handle dtype, const nonstd::optional<std::string>& device_id) {
              return Empty(ToShape(shape), internal::GetDtype(dtype), GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("empty",
          [](py::tuple shape, py::handle dtype, Device& device) {
              return Empty(ToShape(shape), internal::GetDtype(dtype), device).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"));
    m.def("full",
          [](py::tuple shape, Scalar fill_value, py::handle dtype, const nonstd::optional<std::string>& device_id) {
              return Full(ToShape(shape), fill_value, internal::GetDtype(dtype), GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("full",
          [](py::tuple shape, Scalar fill_value, py::handle dtype, Device& device) {
              return Full(ToShape(shape), fill_value, internal::GetDtype(dtype), device).move_body();
          },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("dtype"),
          py::arg("device"));
    m.def("full",
          [](py::tuple shape, Scalar fill_value, const nonstd::optional<std::string>& device_id) {
              return Full(ToShape(shape), fill_value, GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("device") = nullptr);
    m.def("full",
          [](py::tuple shape, Scalar fill_value, Device& device) { return Full(ToShape(shape), fill_value, device).move_body(); },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("device"));
    m.def("zeros",
          [](py::tuple shape, py::handle dtype, const nonstd::optional<std::string>& device_id) {
              return Zeros(ToShape(shape), internal::GetDtype(dtype), GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("zeros",
          [](py::tuple shape, py::handle dtype, Device& device) {
              return Zeros(ToShape(shape), internal::GetDtype(dtype), device).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"));
    m.def("ones",
          [](py::tuple shape, py::handle dtype, const nonstd::optional<std::string>& device_id) {
              return Ones(ToShape(shape), internal::GetDtype(dtype), GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("ones",
          [](py::tuple shape, py::handle dtype, Device& device) {
              return Ones(ToShape(shape), internal::GetDtype(dtype), device).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device"));
    m.def("arange",
          [](Scalar start,
             const nonstd::optional<Scalar>& stop,
             const nonstd::optional<Scalar>& step,
             py::handle dtype,
             const nonstd::optional<std::string>& device_id) { return MakeArangeArray(start, stop, step, dtype, GetDevice(device_id)); },
          py::arg("start"),
          py::arg("stop") = nullptr,
          py::arg("step") = nullptr,
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("arange",
          [](Scalar start, const nonstd::optional<Scalar>& stop, const nonstd::optional<Scalar>& step, py::handle dtype, Device& device) {
              return MakeArangeArray(start, stop, step, dtype, device);
          },
          py::arg("start"),
          py::arg("stop") = nullptr,
          py::arg("step") = nullptr,
          py::arg("dtype") = nullptr,
          py::arg("device"));
    m.def("empty_like",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::string>& device_id) {
              return EmptyLike(Array{a}, GetDevice(device_id)).move_body();
          },
          py::arg("a"),
          py::arg("device") = nullptr);
    m.def("empty_like",
          [](const ArrayBodyPtr& a, Device& device) { return EmptyLike(Array{a}, device).move_body(); },
          py::arg("a"),
          py::arg("device"));
    m.def("full_like",
          [](const ArrayBodyPtr& a, Scalar value, const nonstd::optional<std::string>& device_id) {
              return FullLike(Array{a}, value, GetDevice(device_id)).move_body();
          },
          py::arg("a"),
          py::arg("fill_value"),
          py::arg("device") = nullptr);
    m.def("full_like",
          [](const ArrayBodyPtr& a, Scalar value, Device& device) { return FullLike(Array{a}, value, device).move_body(); },
          py::arg("a"),
          py::arg("fill_value"),
          py::arg("device"));
    m.def("zeros_like",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::string>& device_id) {
              return ZerosLike(Array{a}, GetDevice(device_id)).move_body();
          },
          py::arg("a"),
          py::arg("device") = nullptr);
    m.def("zeros_like",
          [](const ArrayBodyPtr& a, Device& device) { return ZerosLike(Array{a}, device).move_body(); },
          py::arg("a"),
          py::arg("device"));
    m.def("ones_like",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::string>& device_id) {
              return OnesLike(Array{a}, GetDevice(device_id)).move_body();
          },
          py::arg("a"),
          py::arg("device") = nullptr);
    m.def("ones_like",
          [](const ArrayBodyPtr& a, Device& device) { return OnesLike(Array{a}, device).move_body(); },
          py::arg("a"),
          py::arg("device"));
    m.def("copy", [](const ArrayBodyPtr& a) { return Copy(Array{a}).move_body(); }, py::arg("a"));

    // indexing routines
    m.def("take",
          [](const ArrayBodyPtr& a, const ArrayBodyPtr& indices, const nonstd::optional<int8_t>& axis) {
              if (!axis.has_value()) {
                  throw NotImplementedError("axis=None is not yet supported for xchainer.take.");
              }
              return Take(Array{a}, Array{indices}, axis.value()).move_body();
          },
          py::arg("a"),
          py::arg("indices"),
          py::arg("axis") = nullptr);

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
          [](const ArrayBodyPtr& a, const nonstd::optional<NdimVector<int8_t>>& axis) { return Squeeze(Array{a}, axis).move_body(); },
          py::arg("a"),
          py::arg("axis") = nullptr);
    m.def("squeeze",
          [](const ArrayBodyPtr& a, int8_t axis) { return Squeeze(Array{a}, NdimVector<int8_t>{axis}).move_body(); },
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
    m.def("negative", [](const ArrayBodyPtr& x) { return Negative(Array{x}).move_body(); }, py::arg("x"));
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
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return Sum(Array{a}, NdimVector<int8_t>{axis}, keepdims).move_body(); },
          py::arg("a"),
          py::arg("axis"),
          py::arg("keepdims") = false);
    m.def("sum",
          [](const ArrayBodyPtr& a, const nonstd::optional<NdimVector<int8_t>>& axis, bool keepdims) {
              return Sum(Array{a}, axis, keepdims).move_body();
          },
          py::arg("a"),
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    m.def("maximum", [](const ArrayBodyPtr& x1, Scalar x2) { return Maximum(Array{x1}, x2).move_body(); }, py::arg("x1"), py::arg("x2"));
    m.def("maximum", [](Scalar x1, const ArrayBodyPtr& x2) { return Maximum(x1, Array{x2}).move_body(); }, py::arg("x1"), py::arg("x2"));
    m.def("exp", [](const ArrayBodyPtr& x) { return Exp(Array{x}).move_body(); }, py::arg("x"));
    m.def("log", [](const ArrayBodyPtr& x) { return Log(Array{x}).move_body(); }, py::arg("x"));
    m.def("logsumexp",
          [](const ArrayBodyPtr& x, int8_t axis, bool keepdims) {
              return LogSumExp(Array{x}, NdimVector<int8_t>{axis}, keepdims).move_body();
          },
          py::arg("x"),
          py::arg("axis"),
          py::arg("keepdims") = false);
    m.def("logsumexp",
          [](const ArrayBodyPtr& x, const nonstd::optional<NdimVector<int8_t>>& axis, bool keepdims) {
              return LogSumExp(Array{x}, axis, keepdims).move_body();
          },
          py::arg("x"),
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    m.def("log_softmax",
          [](const ArrayBodyPtr& x, int8_t axis) { return LogSoftmax(Array{x}, NdimVector<int8_t>{axis}).move_body(); },
          py::arg("x"),
          py::arg("axis"));
    m.def("log_softmax",
          [](const ArrayBodyPtr& x, const nonstd::optional<NdimVector<int8_t>>& axis) { return LogSoftmax(Array{x}, axis).move_body(); },
          py::arg("x"),
          py::arg("axis") = nullptr);

    // sorting routines
    m.def("argmax",
          [](const ArrayBodyPtr& a, const nonstd::optional<int8_t>& axis) { return ArgMax(Array{a}, axis).move_body(); },
          py::arg("a"),
          py::arg("axis") = nullptr);

    // statistics routines
    m.def("amax",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return AMax(Array{a}, NdimVector<int8_t>{axis}, keepdims).move_body(); },
          py::arg("a"),
          py::arg("axis"),
          py::arg("keepdims") = false);
    m.def("amax",
          [](const ArrayBodyPtr& a, const nonstd::optional<NdimVector<int8_t>>& axis, bool keepdims) {
              return AMax(Array{a}, axis, keepdims).move_body();
          },
          py::arg("a"),
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    m.attr("max") = m.attr("amax");
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
