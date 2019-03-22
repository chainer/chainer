#include "chainerx/python/routines.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/constant.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/normalization.h"
#include "chainerx/routines/pooling.h"
#include "chainerx/routines/sorting.h"
#include "chainerx/scalar.h"
#include "chainerx/stack_vector.h"

#include "chainerx/python/array.h"
#include "chainerx/python/array_index.h"
#include "chainerx/python/axes.h"
#include "chainerx/python/common.h"
#include "chainerx/python/device.h"
#include "chainerx/python/dtype.h"
#include "chainerx/python/shape.h"
#include "chainerx/python/stack_vector.h"
#include "chainerx/python/strides.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

namespace {

using internal::MoveArrayBodies;
using internal::MoveArrayBody;

ArrayBodyPtr MakeArrayFromBuffer(py::buffer buffer, py::handle dtype, int64_t count, int64_t offset, py::handle device) {
    const py::buffer_info& info = buffer.request();

    int64_t n_bytes = info.size * info.itemsize;
    if (offset < 0 || offset > n_bytes) {
        throw ChainerxError{"offset must be non-negative and no greater than buffer length (", n_bytes, ")"};
    }

    if (!internal::IsContiguous(
                Shape{info.shape.begin(), info.shape.end()}, Strides{info.strides.begin(), info.strides.end()}, info.itemsize)) {
        throw ChainerxError{"ndarray is not C-contiguous"};
    }

    n_bytes -= offset;
    if (count < 0) {
        if (n_bytes % info.itemsize != 0) {
            throw ChainerxError{"buffer size must be a multiple of element size"};
        }
        count = n_bytes / info.itemsize;
    } else if (n_bytes < count * info.itemsize) {
        throw ChainerxError{"buffer is smaller than requested size"};
    }

    Shape shape{count};
    std::shared_ptr<void> data{info.ptr, [](void*) {}};

    return MoveArrayBody(chainerx::FromData(shape, GetDtype(dtype), data, nonstd::nullopt, offset, GetDevice(device)));
}

void InitChainerxCreation(pybind11::module& m) {
    // creation routines
    // TODO(niboshi): Accept CuPy ndarray in `array` and `asarray`. In principle it's CuPy's responsibility to provide some standard
    // interface to allow this, but users may want to convert cupy.ndarray to ChainerX before CuPy's support will be implemented. In such
    // case, ChainerX should provide the support for convenience.
    // TODO(niboshi): Add convenient function to convert to CuPy ndarray. Currently chainerx.ndarray exposes internal pointer
    // (ndarray.data_ptr, etc.) to support this, but users may want more convenient method. In principle ChainerX should support some
    // standard way (not depending on CuPy), but we might tentatively provide one which concretely depends on CuPy.
    m.def("array",
          [](py::handle object, py::handle dtype, bool copy, py::handle device) { return MakeArray(object, dtype, copy, device); },
          py::arg("object"),
          py::arg("dtype") = nullptr,
          py::arg("copy") = true,
          py::arg("device") = nullptr);
    // TODO(niboshi): Rename `object` to `a` as per numpy.
    m.def("asarray",
          [](py::handle object, py::handle dtype, py::handle device) { return MakeArray(object, dtype, false, device); },
          py::arg("object"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("ascontiguousarray",
          [](py::handle a, py::handle dtype, py::handle device) {
              Array arr{MakeArray(a, dtype, false, device)};
              return MoveArrayBody(AsContiguousArray(arr));
          },
          py::arg("a"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("empty",
          [](py::tuple shape, py::handle dtype, py::handle device) {
              return MoveArrayBody(Empty(ToShape(shape), dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          py::arg("shape"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("empty",
          [](py::int_ dim, py::handle dtype, py::handle device) {
              return MoveArrayBody(Empty(Shape{dim}, dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          py::arg("shape"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("full",
          [](py::tuple shape, Scalar fill_value, py::handle dtype, py::handle device) {
              return MoveArrayBody(Full(ToShape(shape), fill_value, GetDtype(dtype), GetDevice(device)));
          },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("full",
          [](py::int_ dim, Scalar fill_value, py::handle dtype, py::handle device) {
              return MoveArrayBody(Full(Shape{dim}, fill_value, GetDtype(dtype), GetDevice(device)));
          },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("full",
          [](py::tuple shape, Scalar fill_value, py::handle device) {
              return MoveArrayBody(Full(ToShape(shape), fill_value, GetDevice(device)));
          },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("device") = nullptr);
    m.def("full",
          [](py::int_ dim, Scalar fill_value, py::handle device) { return MoveArrayBody(Full(Shape{dim}, fill_value, GetDevice(device))); },
          py::arg("shape"),
          py::arg("fill_value"),
          py::arg("device") = nullptr);
    m.def("zeros",
          [](py::tuple shape, py::handle dtype, py::handle device) {
              return MoveArrayBody(Zeros(ToShape(shape), dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          py::arg("shape"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("zeros",
          [](py::int_ dim, py::handle dtype, py::handle device) {
              return MoveArrayBody(Zeros(Shape{dim}, dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          py::arg("shape"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("ones",
          [](py::tuple shape, py::handle dtype, py::handle device) {
              return MoveArrayBody(Ones(ToShape(shape), dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          py::arg("shape"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("ones",
          [](py::int_ dim, py::handle dtype, py::handle device) {
              return MoveArrayBody(Ones(Shape{dim}, dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          py::arg("shape"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("arange",
          [](Scalar start_or_stop,
             const nonstd::optional<Scalar>& maybe_stop,
             const nonstd::optional<Scalar>& maybe_step,
             py::handle dtype,
             py::handle device) {
              DtypeKind start_or_stop_dtype_kind = start_or_stop.kind();
              Scalar start{0, start_or_stop_dtype_kind};
              Scalar stop{start_or_stop};
              Scalar step = maybe_step.has_value() ? maybe_step.value() : Scalar{1, start_or_stop_dtype_kind};

              if (maybe_stop.has_value()) {
                  start = start_or_stop;
                  stop = maybe_stop.value();
              }

              return dtype.is_none() ? MoveArrayBody(Arange(start, stop, step, GetDevice(device)))
                                     : MoveArrayBody(Arange(start, stop, step, GetDtype(dtype), GetDevice(device)));
          },
          py::arg("start"),
          py::arg("stop") = nullptr,
          py::arg("step") = nullptr,
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("empty_like",
          [](const ArrayBodyPtr& a, py::handle device) { return MoveArrayBody(EmptyLike(Array{a}, GetDevice(device))); },
          py::arg("a"),
          py::arg("device") = nullptr);
    m.def("full_like",
          [](const ArrayBodyPtr& a, Scalar value, py::handle device) {
              return MoveArrayBody(FullLike(Array{a}, value, GetDevice(device)));
          },
          py::arg("a"),
          py::arg("fill_value"),
          py::arg("device") = nullptr);
    m.def("zeros_like",
          [](const ArrayBodyPtr& a, py::handle device) { return MoveArrayBody(ZerosLike(Array{a}, GetDevice(device))); },
          py::arg("a"),
          py::arg("device") = nullptr);
    m.def("ones_like",
          [](const ArrayBodyPtr& a, py::handle device) { return MoveArrayBody(OnesLike(Array{a}, GetDevice(device))); },
          py::arg("a"),
          py::arg("device") = nullptr);
    m.def("copy", [](const ArrayBodyPtr& a) { return MoveArrayBody(Copy(Array{a})); }, py::arg("a"));
    m.def("frombuffer",
          &MakeArrayFromBuffer,
          py::arg("buffer"),
          py::arg("dtype") = "float32",
          py::arg("count") = -1,
          py::arg("offset") = 0,
          py::arg("device") = nullptr);
    m.def("identity",
          [](int64_t n, py::handle dtype, py::handle device) {
              return MoveArrayBody(Identity(n, dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          py::arg("n"),
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
    m.def("eye",
          [](int64_t n, nonstd::optional<int64_t> m, int64_t k, py::handle dtype, py::handle device) {
              if (!m.has_value()) {
                  m = n;
              }
              return MoveArrayBody(Eye(n, m.value(), k, GetDtype(dtype), GetDevice(device)));
          },
          py::arg("N"),
          py::arg("M") = nullptr,
          py::arg("k") = 0,
          py::arg("dtype") = "float64",
          py::arg("device") = nullptr);
    m.def("diag",
          [](const ArrayBodyPtr& v, int64_t k, py::handle device) { return MoveArrayBody(Diag(Array{v}, k, GetDevice(device))); },
          py::arg("v"),
          py::arg("k") = 0,
          py::arg("device") = nullptr);
    m.def("diagflat",
          [](const ArrayBodyPtr& v, int64_t k, py::handle device) { return MoveArrayBody(Diagflat(Array{v}, k, GetDevice(device))); },
          py::arg("v"),
          py::arg("k") = 0,
          py::arg("device") = nullptr);
    m.def("linspace",
          [](Scalar start, Scalar stop, int64_t num, bool endpoint, py::handle dtype, py::handle device) {
              return MoveArrayBody(Linspace(
                      start,
                      stop,
                      num,
                      endpoint,
                      dtype.is_none() ? nonstd::optional<Dtype>{nonstd::nullopt} : nonstd::optional<Dtype>{GetDtype(dtype)},
                      GetDevice(device)));
          },
          py::arg("start"),
          py::arg("stop"),
          py::arg("num") = 50,
          py::arg("endpoint") = true,
          py::arg("dtype") = nullptr,
          py::arg("device") = nullptr);
}

void InitChainerxIndexing(pybind11::module& m) {
    // indexing routines
    m.def("take",
          [](const ArrayBodyPtr& a, py::handle indices, const nonstd::optional<int8_t>& axis) {
              if (!axis.has_value()) {
                  throw NotImplementedError{"axis=None is not yet supported for chainerx.take."};
              }
              if (py::isinstance<ArrayBody>(indices)) {
                  return MoveArrayBody(Take(Array{a}, Array{py::cast<ArrayBodyPtr>(indices)}, axis.value()));
              }
              if (py::isinstance<py::sequence>(indices)) {
                  nonstd::optional<Dtype> dtype = Dtype::kInt64;
                  return MoveArrayBody(Take(Array{a}, Array{MakeArray(indices, dtype, false, a->device())}, axis.value()));
              }
              if (py::isinstance<py::array>(indices)) {
                  return MoveArrayBody(
                          Take(Array{a}, Array{MakeArrayFromNumpyArray(py::cast<py::array>(indices), a->device())}, axis.value()));
              }
              throw py::type_error{"only integers, slices (`:`), sequence, numpy.ndarray and chainerx.newaxis (`None`) are valid indices"};
          },
          py::arg("a"),
          py::arg("indices"),
          py::arg("axis"));
}

void InitChainerxLinalg(pybind11::module& m) {
    // linalg routines
    m.def("dot",
          [](const ArrayBodyPtr& a, const ArrayBodyPtr& b) { return MoveArrayBody(Dot(Array{a}, Array{b})); },
          py::arg("a"),
          py::arg("b"));
}

void InitChainerxLogic(pybind11::module& m) {
    // logic routines
    m.def("equal",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Equal(Array{x1}, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("not_equal",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(NotEqual(Array{x1}, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("greater",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Greater(Array{x1}, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("greater_equal",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(GreaterEqual(Array{x1}, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("less",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Less(Array{x1}, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("less_equal",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(LessEqual(Array{x1}, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("logical_not", [](const ArrayBodyPtr& x) { return MoveArrayBody(LogicalNot(Array{x})); }, py::arg("x"));
}

void InitChainerxManipulation(pybind11::module& m) {
    // manipulation routines
    m.def("transpose",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::vector<int8_t>>& axes) {
              return MoveArrayBody(Transpose(Array{a}, ToAxes(axes)));
          },
          py::arg("a"),
          py::arg("axes") = nullptr);
    m.def("transpose",
          [](const ArrayBodyPtr& a, int8_t axes) { return MoveArrayBody(Transpose(Array{a}, {axes})); },
          py::arg("a"),
          py::arg("axes") = nullptr);
    m.def("reshape",
          [](const ArrayBodyPtr& a, py::tuple newshape) { return MoveArrayBody(Reshape(Array{a}, ToShape(newshape))); },
          py::arg("a"),
          py::arg("newshape"));
    m.def("reshape",
          [](const ArrayBodyPtr& a, const std::vector<int64_t>& newshape) {
              return MoveArrayBody(Reshape(Array{a}, {newshape.begin(), newshape.end()}));
          },
          py::arg("a"),
          py::arg("newshape"));
    m.def("reshape",
          [](const ArrayBodyPtr& a, py::args args) {
              if (args.size() == 0) {
                  throw ChainerxError("Reshape is missing shape argument.");
              }
              return MoveArrayBody(Reshape(Array{a}, ToShape(args)));
          },
          py::arg("a"));
    m.def("squeeze",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(Squeeze(Array{a}, ToAxes(axis)));
          },
          py::arg("a"),
          py::arg("axis") = nullptr);
    m.def("squeeze",
          [](const ArrayBodyPtr& a, int8_t axis) { return MoveArrayBody(Squeeze(Array{a}, Axes{axis})); },
          py::arg("a"),
          py::arg("axis"));
    m.def("broadcast_to",
          [](const ArrayBodyPtr& array, py::tuple shape) { return MoveArrayBody(Array{array}.BroadcastTo(ToShape(shape))); },
          py::arg("array"),
          py::arg("shape"));
    m.def("concatenate",
          [](py::sequence arrays, nonstd::optional<int8_t> axis) {
              std::vector<Array> xs;
              xs.reserve(arrays.size());
              std::transform(arrays.begin(), arrays.end(), std::back_inserter(xs), [](const auto& item) {
                  return Array{py::cast<ArrayBodyPtr>(item)};
              });
              return MoveArrayBody(Concatenate(xs, axis));
          },
          py::arg("arrays"),
          py::arg("axis") = 0);
    m.def("stack",
          [](py::sequence arrays, int8_t axis) {
              std::vector<Array> xs;
              xs.reserve(arrays.size());
              std::transform(arrays.begin(), arrays.end(), std::back_inserter(xs), [](const auto& item) {
                  return Array{py::cast<ArrayBodyPtr>(item)};
              });
              return MoveArrayBody(Stack(xs, axis));
          },
          py::arg("arrays"),
          py::arg("axis") = 0);
    m.def("split",
          [](const ArrayBodyPtr& ary, py::handle indices_or_sections, int8_t axis) {
              // TODO(niboshi): Perhaps we would want more general approach to handle multi-type arguments like indices_or_sections to
              // provide more helpful error message for users.

              auto split_sections = [](const ArrayBodyPtr& ary, int64_t sections, int8_t axis) {
                  return MoveArrayBodies(Split(Array{ary}, sections, axis));
              };
              auto split_indices = [](const ArrayBodyPtr& ary, const std::vector<int64_t>& indices, int8_t axis) {
                  return MoveArrayBodies(Split(Array{ary}, indices, axis));
              };

              // Converts an python float to sections (int64_t).
              // Raises ValueError if the value has non-zero fraction.
              auto pyfloat_to_sections_or_value_error = [](py::handle num) {
                  CHAINERX_ASSERT(py::isinstance<py::float_>(num));
                  double num_fp = py::cast<double>(num);
                  auto num_int = static_cast<int64_t>(num_fp);
                  if (static_cast<double>(num_int) != num_fp) {
                      throw py::value_error{"Sections must be an integer."};
                  }
                  return num_int;
              };

              // sections: int
              if (py::isinstance<py::int_>(indices_or_sections)) {
                  int64_t sections = py::cast<int64_t>(indices_or_sections);
                  return split_sections(ary, sections, axis);
              }
              // sections: float
              if (py::isinstance<py::float_>(indices_or_sections)) {
                  int64_t sections = pyfloat_to_sections_or_value_error(indices_or_sections);
                  return split_sections(ary, sections, axis);
              }
              // numpy.ndarray
              if (py::isinstance<py::array>(indices_or_sections)) {
                  py::array np_ios = py::cast<py::array>(indices_or_sections);
                  if (np_ios.ndim() >= 2) {
                      throw py::value_error{std::string{"Too many dimensions of indices: "} + std::to_string(np_ios.ndim())};
                  }
                  // sections: scalar
                  if (np_ios.ndim() == 0) {
                      int64_t sections{};
                      py::object scalar_np = np_ios.attr("tolist")();
                      if (py::isinstance<py::int_>(scalar_np)) {
                          sections = py::cast<int64_t>(scalar_np);
                      } else if (py::isinstance<py::float_>(scalar_np)) {
                          sections = pyfloat_to_sections_or_value_error(scalar_np);
                      } else {
                          throw py::type_error{"Sections must be an integer."};
                      }
                      return split_sections(ary, sections, axis);
                  }

                  // indices: (0,)-shape
                  if (np_ios.size() == 0) {
                      return split_indices(ary, {}, axis);
                  }

                  if (np_ios.dtype().kind() != 'i') {
                      throw py::type_error{std::string{"Indices must be integers."}};
                  }
                  // indices: non-scalar
                  std::vector<int64_t> indices{};
                  py::list indices_pylist = np_ios.attr("tolist")();
                  for (py::handle item : indices_pylist) {
                      indices.emplace_back(py::cast<int64_t>(item));
                  }

                  return split_indices(ary, indices, axis);
              }
              // indices: sequence
              if (py::isinstance<py::sequence>(indices_or_sections)) {
                  std::vector<int64_t> indices{};
                  try {
                      indices = py::cast<std::vector<int64_t>>(indices_or_sections);
                  } catch (const py::cast_error& e) {
                      throw py::type_error{std::string{"Indices not understood: "} + py::cast<std::string>(py::repr(indices_or_sections))};
                  }

                  return split_indices(ary, indices, axis);
              }
              throw py::type_error{std::string{"indices_or_sections not understood: "} +
                                   py::cast<std::string>(py::repr(indices_or_sections))};
          },
          py::arg("ary"),
          py::arg("indices_or_sections"),
          py::arg("axis") = 0);
}

void InitChainerxMath(pybind11::module& m) {
    // math routines
    m.def("negative", [](const ArrayBodyPtr& x) { return MoveArrayBody(Negative(Array{x})); }, py::arg("x"));
    m.def("add",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Array{x1} + Array{x2}); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("add", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Add(Array{x1}, x2)); }, py::arg("x1"), py::arg("x2"));
    m.def("add", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Add(x1, Array{x2})); }, py::arg("x1"), py::arg("x2"));
    m.def("subtract",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Array{x1} - Array{x2}); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("subtract",
          [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Subtract(Array{x1}, x2)); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("subtract",
          [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Subtract(x1, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("multiply",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Array{x1} * Array{x2}); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("multiply",
          [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Multiply(Array{x1}, x2)); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("multiply",
          [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Multiply(x1, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("divide",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Array{x1} / Array{x2}); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("divide", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Divide(Array{x1}, x2)); }, py::arg("x1"), py::arg("x2"));
    m.def("divide", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Divide(x1, Array{x2})); }, py::arg("x1"), py::arg("x2"));
    m.def("floor_divide",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(FloorDivide(Array{x1}, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("floor_divide",
          [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(FloorDivide(Array{x1}, x2)); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("floor_divide",
          [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(FloorDivide(x1, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("true_divide",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(TrueDivide(Array{x1}, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("true_divide",
          [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(TrueDivide(Array{x1}, x2)); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("true_divide",
          [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(TrueDivide(x1, Array{x2})); },
          py::arg("x1"),
          py::arg("x2"));
    m.def("sum",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(Sum(Array{a}, Axes{axis}, keepdims)); },
          py::arg("a"),
          py::arg("axis"),
          py::arg("keepdims") = false);
    m.def("sum",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Sum(Array{a}, ToAxes(axis), keepdims));
          },
          py::arg("a"),
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    m.def("maximum", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Maximum(Array{x1}, x2)); }, py::arg("x1"), py::arg("x2"));
    m.def("maximum", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Maximum(x1, Array{x2})); }, py::arg("x1"), py::arg("x2"));
    m.def("minimum", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Minimum(Array{x1}, x2)); }, py::arg("x1"), py::arg("x2"));
    m.def("minimum", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Minimum(x1, Array{x2})); }, py::arg("x1"), py::arg("x2"));
    m.def("exp", [](const ArrayBodyPtr& x) { return MoveArrayBody(Exp(Array{x})); }, py::arg("x"));
    m.def("log", [](const ArrayBodyPtr& x) { return MoveArrayBody(Log(Array{x})); }, py::arg("x"));
    m.def("logsumexp",
          [](const ArrayBodyPtr& x, int8_t axis, bool keepdims) { return MoveArrayBody(LogSumExp(Array{x}, Axes{axis}, keepdims)); },
          py::arg("x"),
          py::arg("axis"),
          py::arg("keepdims") = false);
    m.def("logsumexp",
          [](const ArrayBodyPtr& x, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(LogSumExp(Array{x}, ToAxes(axis), keepdims));
          },
          py::arg("x"),
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    m.def("log_softmax",
          [](const ArrayBodyPtr& x, int8_t axis) { return MoveArrayBody(LogSoftmax(Array{x}, Axes{axis})); },
          py::arg("x"),
          py::arg("axis"));
    m.def("log_softmax",
          [](const ArrayBodyPtr& x, const nonstd::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(LogSoftmax(Array{x}, ToAxes(axis)));
          },
          py::arg("x"),
          py::arg("axis") = nullptr);
    m.def("sqrt", [](const ArrayBodyPtr& x) { return MoveArrayBody(Sqrt(Array{x})); }, py::arg("x"));
    m.def("tanh", [](const ArrayBodyPtr& x) { return MoveArrayBody(Tanh(Array{x})); }, py::arg("x"));
    m.def("isnan", [](const ArrayBodyPtr& x) { return MoveArrayBody(IsNan(Array{x})); }, py::arg("x"));
    m.def("isinf", [](const ArrayBodyPtr& x) { return MoveArrayBody(IsInf(Array{x})); }, py::arg("x"));
}

void InitChainerxSorting(pybind11::module& m) {
    // sorting routines
    m.def("argmax",
          [](const ArrayBodyPtr& a, const nonstd::optional<int8_t>& axis) { return MoveArrayBody(ArgMax(Array{a}, ToAxes(axis))); },
          py::arg("a"),
          py::arg("axis") = nullptr);
}

void InitChainerxStatistics(pybind11::module& m) {
    // statistics routines
    m.def("amax",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(AMax(Array{a}, Axes{axis}, keepdims)); },
          py::arg("a"),
          py::arg("axis"),
          py::arg("keepdims") = false);
    m.def("amax",
          [](const ArrayBodyPtr& a, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(AMax(Array{a}, ToAxes(axis), keepdims));
          },
          py::arg("a"),
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    m.attr("max") = m.attr("amax");
}

void InitChainerxConnection(pybind11::module& m) {
    // connection routines
    m.def("conv",
          [](const ArrayBodyPtr& x,
             const ArrayBodyPtr& w,
             const nonstd::optional<ArrayBodyPtr>& b,
             py::handle stride,
             py::handle pad,
             bool cover_all) {
              // Create an Array from x to compute the image dimensions and the expected number of stride and padding elements.
              Array x_array{x};
              int8_t ndim = x_array.ndim() - 2;
              return MoveArrayBody(
                      Conv(x_array,
                           Array{w},
                           b.has_value() ? nonstd::optional<Array>{Array{*b}} : nonstd::nullopt,
                           ToStackVector<int64_t>(stride, ndim),
                           ToStackVector<int64_t>(pad, ndim),
                           cover_all));
          },
          py::arg("x"),
          py::arg("w"),
          py::arg("b") = nullptr,
          py::arg("stride") = 1,
          py::arg("pad") = 0,
          py::arg("cover_all") = false);
    m.def("conv_transpose",
          [](const ArrayBodyPtr& x,
             const ArrayBodyPtr& w,
             const nonstd::optional<ArrayBodyPtr>& b,
             py::handle stride,
             py::handle pad,
             const nonstd::optional<py::tuple>& outsize) {
              // Create an Array from x to compute the image dimensions and the expected number of stride and padding elements.
              Array x_array{x};
              int8_t ndim = x_array.ndim() - 2;
              return MoveArrayBody(ConvTranspose(
                      x_array,
                      Array{w},
                      b.has_value() ? nonstd::optional<Array>{Array{*b}} : nonstd::nullopt,
                      ToStackVector<int64_t>(stride, ndim),
                      ToStackVector<int64_t>(pad, ndim),
                      outsize.has_value() ? nonstd::optional<StackVector<int64_t, kMaxNdim>>{ToStackVector<int64_t>(*outsize, ndim)}
                                          : nonstd::nullopt));
          },
          py::arg("x"),
          py::arg("w"),
          py::arg("b") = nullptr,
          py::arg("stride") = 1,
          py::arg("pad") = 0,
          py::arg("outsize") = nullptr);
    m.def("linear",
          [](const ArrayBodyPtr& x, const ArrayBodyPtr& w, const nonstd::optional<ArrayBodyPtr>& b, int8_t n_batch_axes) {
              return MoveArrayBody(
                      Linear(Array{x}, Array{w}, b.has_value() ? nonstd::optional<Array>{Array{*b}} : nonstd::nullopt, n_batch_axes));
          },
          py::arg("x"),
          py::arg("w"),
          py::arg("b") = nullptr,
          py::arg("n_batch_axes") = 1);
}

void InitChainerxNormalization(pybind11::module& m) {
    // normalization routines
    m.def("batch_norm",
          [](const ArrayBodyPtr& x,
             const ArrayBodyPtr& gamma,
             const ArrayBodyPtr& beta,
             const ArrayBodyPtr& running_mean,
             const ArrayBodyPtr& running_var,
             Scalar eps,
             Scalar decay,
             const nonstd::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(
                      BatchNorm(Array{x}, Array{gamma}, Array{beta}, Array{running_mean}, Array{running_var}, eps, decay, ToAxes(axis)));
          },
          py::arg("x"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps") = 2e-5,
          py::arg("decay") = 0.9,
          py::arg("axis") = nullptr);
    m.def("fixed_batch_norm",
          [](const ArrayBodyPtr& x,
             const ArrayBodyPtr& gamma,
             const ArrayBodyPtr& beta,
             const ArrayBodyPtr& mean,
             const ArrayBodyPtr& var,
             Scalar eps,
             const nonstd::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(FixedBatchNorm(Array{x}, Array{gamma}, Array{beta}, Array{mean}, Array{var}, eps, ToAxes(axis)));
          },
          py::arg("x"),
          py::arg("gamma"),
          py::arg("beta"),
          py::arg("mean"),
          py::arg("var"),
          py::arg("eps") = 2e-5,
          py::arg("axis") = nullptr);
}

void InitChainerxPooling(pybind11::module& m) {
    // pooling routines
    // TODO(sonots): Support return_indicies option of chainer.functions.max_pooling_nd.
    m.def("max_pool",
          [](const ArrayBodyPtr& x, py::handle ksize, py::handle stride, py::handle pad, bool cover_all) {
              Array x_array{x};
              int8_t ndim = x_array.ndim() - 2;
              return MoveArrayBody(
                      MaxPool(x_array,
                              ToStackVector<int64_t>(ksize, ndim),
                              stride.is_none() ? ToStackVector<int64_t>(ksize, ndim) : ToStackVector<int64_t>(stride, ndim),
                              ToStackVector<int64_t>(pad, ndim),
                              cover_all));
          },
          py::arg("x"),
          py::arg("ksize"),
          py::arg("stride") = py::none(),
          py::arg("pad") = 0,
          py::arg("cover_all") = false);
    m.def("average_pool",
          [](const ArrayBodyPtr& x, py::handle ksize, py::handle stride, py::handle pad, const std::string& pad_mode) {
              Array x_array{x};
              int8_t ndim = x_array.ndim() - 2;

              AveragePoolPadMode mode{};
              if (pad_mode == "zero") {
                  mode = AveragePoolPadMode::kZero;
              } else if (pad_mode == "ignore") {
                  mode = AveragePoolPadMode::kIgnore;
              } else {
                  throw py::value_error{"pad_mode must be either of 'zero' or 'ignore'"};
              }

              return MoveArrayBody(AveragePool(
                      x_array,
                      ToStackVector<int64_t>(ksize, ndim),
                      stride.is_none() ? ToStackVector<int64_t>(ksize, ndim) : ToStackVector<int64_t>(stride, ndim),
                      ToStackVector<int64_t>(pad, ndim),
                      mode));
          },
          py::arg("x"),
          py::arg("ksize"),
          py::arg("stride") = py::none(),
          py::arg("pad") = 0,
          py::arg("pad_mode") = "ignore");
}

}  // namespace

void InitChainerxRoutines(pybind11::module& m) {
    InitChainerxCreation(m);
    InitChainerxIndexing(m);
    InitChainerxLinalg(m);
    InitChainerxLogic(m);
    InitChainerxManipulation(m);
    InitChainerxMath(m);
    InitChainerxSorting(m);
    InitChainerxStatistics(m);
    InitChainerxConnection(m);
    InitChainerxNormalization(m);
    InitChainerxPooling(m);
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
