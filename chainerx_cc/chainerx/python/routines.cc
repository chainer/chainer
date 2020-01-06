#include "chainerx/python/common_export.h"

#include "chainerx/python/routines.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/constant.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/dims.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/routines/activation.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/binary.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/evaluation.h"
#include "chainerx/routines/explog.h"
#include "chainerx/routines/hyperbolic.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/logic.h"
#include "chainerx/routines/loss.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/normalization.h"
#include "chainerx/routines/pooling.h"
#include "chainerx/routines/reduction.h"
#include "chainerx/routines/rounding.h"
#include "chainerx/routines/sorting.h"
#include "chainerx/routines/statistics.h"

#include "chainerx/routines/n_step_rnn.h"
#include "chainerx/routines/trigonometric.h"
#include "chainerx/scalar.h"

#include "chainerx/python/array.h"
#include "chainerx/python/array_index.h"
#include "chainerx/python/axes.h"
#include "chainerx/python/common.h"
#include "chainerx/python/device.h"
#include "chainerx/python/dtype.h"
#include "chainerx/python/kwarg.h"
#include "chainerx/python/shape.h"
#include "chainerx/python/stack_vector.h"
#include "chainerx/python/strides.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;
using py::literals::operator""_a;

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

    return MoveArrayBody(chainerx::FromData(shape, GetDtype(dtype), data, absl::nullopt, offset, GetDevice(device)));
}

std::vector<Array> ArrayBodiesToArrays(std::vector<ArrayBodyPtr> array_bodies) {
    std::vector<Array> arrays{};
    arrays.reserve(array_bodies.size());
    for (ArrayBodyPtr& array_body : array_bodies) {
        arrays.emplace_back(std::move(array_body));
    }
    return arrays;
}

CastingMode ParseCastingMode(const std::string& casting) {
    CastingMode mode{};
    if (casting == "no") {
        mode = CastingMode::kNo;
    } else if (casting == "equiv") {
        throw NotImplementedError{"'equiv' casting is not yet implemented."};
    } else if (casting == "safe") {
        throw NotImplementedError{"'safe' casting is not yet implemented."};
    } else if (casting == "same_kind") {
        throw NotImplementedError{"'same_kind' casting is not yet implemented."};
    } else if (casting == "unsafe") {
        throw NotImplementedError{"'unsafe' casting is not yet implemented."};
    } else {
        throw py::value_error{"Casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'."};
    }
    return mode;
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
          "object"_a,
          "dtype"_a = nullptr,
          "copy"_a = true,
          "device"_a = nullptr);
    // TODO(niboshi): Rename `object` to `a` as per numpy.
    m.def("asarray",
          [](py::handle object, py::handle dtype, py::handle device) { return MakeArray(object, dtype, false, device); },
          "object"_a,
          "dtype"_a = nullptr,
          "device"_a = nullptr);
    m.def("ascontiguousarray",
          [](const ArrayBodyPtr& a, py::handle dtype) {
              Array arr{a};
              return MoveArrayBody(AsContiguousArray(arr, dtype.is_none() ? arr.dtype() : GetDtype(dtype)));
          },
          "a"_a,
          "dtype"_a = nullptr);
    m.def("empty",
          [](py::handle shape, py::handle dtype, py::handle device) {
              return MoveArrayBody(Empty(ToShape(shape), dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          "shape"_a,
          "dtype"_a = nullptr,
          "device"_a = nullptr);
    m.def("full",
          [](py::handle shape, Scalar fill_value, py::handle dtype, py::handle device) {
              return MoveArrayBody(Full(ToShape(shape), fill_value, GetDtype(dtype), GetDevice(device)));
          },
          "shape"_a,
          "fill_value"_a,
          "dtype"_a,
          "device"_a = nullptr);
    m.def("full",
          [](py::int_ dim, Scalar fill_value, py::handle dtype, py::handle device) {
              return MoveArrayBody(Full(Shape{dim}, fill_value, GetDtype(dtype), GetDevice(device)));
          },
          "shape"_a,
          "fill_value"_a,
          "dtype"_a,
          "device"_a = nullptr);
    m.def("full",
          [](py::handle shape, Scalar fill_value, py::handle device) {
              return MoveArrayBody(Full(ToShape(shape), fill_value, GetDevice(device)));
          },
          "shape"_a,
          "fill_value"_a,
          "device"_a = nullptr);
    m.def("full",
          [](py::int_ dim, Scalar fill_value, py::handle device) { return MoveArrayBody(Full(Shape{dim}, fill_value, GetDevice(device))); },
          "shape"_a,
          "fill_value"_a,
          "device"_a = nullptr);
    m.def("zeros",
          [](py::handle shape, py::handle dtype, py::kwargs kwargs) {
              py::handle device;
              std::tie(device) = GetKwargs(kwargs, "device");
              return MoveArrayBody(Zeros(ToShape(shape), dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          "shape"_a,
          "dtype"_a = nullptr);
    m.def("zeros",
          [](py::int_ dim, py::handle dtype, py::kwargs kwargs) {
              py::handle device;
              std::tie(device) = GetKwargs(kwargs, "device");
              return MoveArrayBody(Zeros(Shape{dim}, dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          "shape"_a,
          "dtype"_a = nullptr);
    m.def("ones",
          [](py::handle shape, py::handle dtype, py::handle device) {
              return MoveArrayBody(Ones(ToShape(shape), dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          "shape"_a,
          "dtype"_a = nullptr,
          "device"_a = nullptr);
    m.def("ones",
          [](py::int_ dim, py::handle dtype, py::handle device) {
              return MoveArrayBody(Ones(Shape{dim}, dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          "shape"_a,
          "dtype"_a = nullptr,
          "device"_a = nullptr);
    m.def("arange",
          [](Scalar start_or_stop,
             absl::optional<Scalar> maybe_stop,
             absl::optional<Scalar> maybe_step,
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
          "start"_a,
          "stop"_a = nullptr,
          "step"_a = nullptr,
          "dtype"_a = nullptr,
          "device"_a = nullptr);
    m.def("empty_like",
          [](const ArrayBodyPtr& a, py::handle device) { return MoveArrayBody(EmptyLike(Array{a}, GetDevice(device))); },
          "a"_a,
          "device"_a = nullptr);
    m.def("full_like",
          [](const ArrayBodyPtr& a, Scalar value, py::handle device) {
              return MoveArrayBody(FullLike(Array{a}, value, GetDevice(device)));
          },
          "a"_a,
          "fill_value"_a,
          "device"_a = nullptr);
    m.def("zeros_like",
          [](const ArrayBodyPtr& a, py::handle device) { return MoveArrayBody(ZerosLike(Array{a}, GetDevice(device))); },
          "a"_a,
          "device"_a = nullptr);
    m.def("ones_like",
          [](const ArrayBodyPtr& a, py::handle device) { return MoveArrayBody(OnesLike(Array{a}, GetDevice(device))); },
          "a"_a,
          "device"_a = nullptr);
    m.def("copy", [](const ArrayBodyPtr& a) { return MoveArrayBody(Copy(Array{a})); }, "a"_a);
    m.def("frombuffer", &MakeArrayFromBuffer, "buffer"_a, "dtype"_a = "float32", "count"_a = -1, "offset"_a = 0, "device"_a = nullptr);
    m.def("identity",
          [](int64_t n, py::handle dtype, py::handle device) {
              return MoveArrayBody(Identity(n, dtype.is_none() ? Dtype::kFloat32 : GetDtype(dtype), GetDevice(device)));
          },
          "n"_a,
          "dtype"_a = nullptr,
          "device"_a = nullptr);
    m.def("eye",
          [](int64_t n, absl::optional<int64_t> m, int64_t k, py::handle dtype, py::handle device) {
              if (!m.has_value()) {
                  m = n;
              }
              return MoveArrayBody(Eye(n, m.value(), k, GetDtype(dtype), GetDevice(device)));
          },
          "N"_a,
          "M"_a = nullptr,
          "k"_a = 0,
          "dtype"_a = "float64",
          "device"_a = nullptr);
    m.def("diag", [](const ArrayBodyPtr& v, int64_t k) { return MoveArrayBody(Diag(Array{v}, k)); }, "v"_a, "k"_a = 0);
    m.def("diagflat", [](const ArrayBodyPtr& v, int64_t k) { return MoveArrayBody(Diagflat(Array{v}, k)); }, "v"_a, "k"_a = 0);
    m.def("linspace",
          [](Scalar start, Scalar stop, int64_t num, bool endpoint, py::handle dtype, py::handle device) {
              return MoveArrayBody(Linspace(
                      start,
                      stop,
                      num,
                      endpoint,
                      dtype.is_none() ? absl::optional<Dtype>{absl::nullopt} : absl::optional<Dtype>{GetDtype(dtype)},
                      GetDevice(device)));
          },
          "start"_a,
          "stop"_a,
          "num"_a = 50,
          "endpoint"_a = true,
          "dtype"_a = nullptr,
          "device"_a = nullptr);
    m.def("tri",
          [](int64_t n, absl::optional<int64_t> m, int64_t k, py::handle dtype, py::handle device) {
              return MoveArrayBody(Tri(n, m, k, GetDtype(dtype), GetDevice(device)));
          },
          "N"_a,
          "M"_a = nullptr,
          "k"_a = 0,
          "dtype"_a = "float32",
          "device"_a = nullptr);
    m.def("tril", [](const ArrayBodyPtr& m, int64_t k) { return MoveArrayBody(Tril(Array{m}, k)); }, "m"_a, "k"_a = 0);
    m.def("triu", [](const ArrayBodyPtr& m, int64_t k) { return MoveArrayBody(Triu(Array{m}, k)); }, "m"_a, "k"_a = 0);
    m.def("meshgrid", [](py::args xi, py::kwargs kwargs) {
        std::vector<Array> xs;
        MeshgridIndexingMode mode{MeshgridIndexingMode::kCartesian};
        xs.reserve(xi.size());
        std::transform(xi.begin(), xi.end(), std::back_inserter(xs), [](const auto& item) { return Array{py::cast<ArrayBodyPtr>(item)}; });
        if (kwargs.size()) {
            if (kwargs.size() != 1 || !kwargs.contains("indexing")) {
                throw ChainerxError{"Only 'indexing' is a valid keyword argument"};
            }
            py::str index = kwargs["indexing"];
            std::string indexing = index;

            if (indexing == "xy") {
                mode = MeshgridIndexingMode::kCartesian;
            } else if (indexing == "ij") {
                mode = MeshgridIndexingMode::kMatrix;
            } else {
                throw ChainerxError{"Indexing can only be 'xy' or 'ij'."};
            }
        }
        return MoveArrayBodies(Meshgrid(xs, mode));
    });
}

void InitChainerxEvaluation(pybind11::module& m) {
    // evaluation routines
    m.def("accuracy",
          [](const ArrayBodyPtr& y, const ArrayBodyPtr& t, absl::optional<int64_t> ignore_label) {
              return MoveArrayBody(Accuracy(Array{y}, Array{t}, ignore_label));
          },
          "y"_a,
          "t"_a,
          "ignore_label"_a = nullptr);
}

void InitChainerxIndexing(pybind11::module& m) {
    // indexing routines
    m.def("take",
          [](const ArrayBodyPtr& a, py::handle indices, absl::optional<int8_t> axis, const absl::optional<std::string>& mode) {
              if (!axis.has_value()) {
                  throw NotImplementedError{"axis=None is not yet supported for chainerx.take."};
              }
              IndexBoundsMode tmode{};
              if (!mode.has_value()) {
                  tmode = IndexBoundsMode::kDefault;
              } else {
                  const std::string& smode = mode.value();
                  if (smode == "raise") {
                      tmode = IndexBoundsMode::kRaise;
                  } else if (smode == "wrap") {
                      tmode = IndexBoundsMode::kWrap;
                  } else if (smode == "clip") {
                      tmode = IndexBoundsMode::kClip;
                  } else {
                      throw py::value_error{"mode must be 'raise', 'wrap', or 'clip'"};
                  }
              }
              if (py::isinstance<ArrayBody>(indices)) {
                  return MoveArrayBody(Take(Array{a}, Array{py::cast<ArrayBodyPtr>(indices)}, axis.value(), tmode));
              }
              if (py::isinstance<py::sequence>(indices)) {
                  absl::optional<Dtype> dtype = Dtype::kInt64;
                  return MoveArrayBody(Take(Array{a}, Array{MakeArray(indices, dtype, false, a->device())}, axis.value(), tmode));
              }
              if (py::isinstance<py::array>(indices)) {
                  return MoveArrayBody(
                          Take(Array{a}, Array{MakeArrayFromNumpyArray(py::cast<py::array>(indices), a->device())}, axis.value(), tmode));
              }
              throw py::type_error{"only integers, slices (`:`), sequence, numpy.ndarray and chainerx.newaxis (`None`) are valid indices"};
          },
          "a"_a,
          "indices"_a,
          "axis"_a,
          "mode"_a = nullptr);
    m.def("where",
          [](const ArrayBodyPtr& condition, const ArrayBodyPtr& x, const ArrayBodyPtr& y) {
              return MoveArrayBody(Where(Array{condition}, Array{x}, Array{y}));
          },
          "condition"_a,
          "x"_a,
          "y"_a);
    m.def("where",
          [](const ArrayBodyPtr& condition, const ArrayBodyPtr& x, Scalar y) {
              return MoveArrayBody(Where(Array{condition}, Array{x}, y));
          },
          "condition"_a,
          "x"_a,
          "y"_a);
    m.def("where",
          [](const ArrayBodyPtr& condition, Scalar x, const ArrayBodyPtr& y) {
              return MoveArrayBody(Where(Array{condition}, x, Array{y}));
          },
          "condition"_a,
          "x"_a,
          "y"_a);
    m.def("where",
          [](const ArrayBodyPtr& condition, Scalar x, Scalar y) { return MoveArrayBody(Where(Array{condition}, x, y)); },
          "condition"_a,
          "x"_a,
          "y"_a);
    m.def("nonzero", [](const ArrayBodyPtr& a) { return ToTuple(Nonzero(Array{a})); }, "a"_a);
}

void InitChainerxLinalg(pybind11::module& m) {
    // linalg routines
    m.def("dot", [](const ArrayBodyPtr& a, const ArrayBodyPtr& b) { return MoveArrayBody(Dot(Array{a}, Array{b})); }, "a"_a, "b"_a);

    pybind11::module mlinalg = m.def_submodule("linalg");
#if CHAINERX_ENABLE_LAPACK
    mlinalg.def("_is_lapack_available", []() -> bool { return true; });
#else
    mlinalg.def("_is_lapack_available", []() -> bool { return false; });
#endif
    mlinalg.def(
            "solve", [](const ArrayBodyPtr& a, const ArrayBodyPtr& b) { return MoveArrayBody(Solve(Array{a}, Array{b})); }, "a"_a, "b"_a);
    mlinalg.def("inv", [](const ArrayBodyPtr& a) { return MoveArrayBody(Inverse(Array{a})); }, "a"_a);
    mlinalg.def(
            "svd",
            [](const ArrayBodyPtr& a, bool full_matrices, bool compute_uv) -> py::object {
                std::tuple<Array, Array, Array> usvt = Svd(Array{a}, full_matrices, compute_uv);
                Array& u = std::get<0>(usvt);
                Array& s = std::get<1>(usvt);
                Array& vt = std::get<2>(usvt);
                if (!compute_uv) {
                    return py::cast(MoveArrayBody(std::move(s)));
                }
                return py::make_tuple(MoveArrayBody(std::move(u)), MoveArrayBody(std::move(s)), MoveArrayBody(std::move(vt)));
            },
            "a"_a,
            "full_matrices"_a = true,
            "compute_uv"_a = true);
    mlinalg.def(
            "pinv",
            [](const ArrayBodyPtr& a, float rcond) { return MoveArrayBody(PseudoInverse(Array{a}, rcond)); },
            "a"_a,
            "rcond"_a = 1e-15);
    mlinalg.def(
            "qr",
            [](const ArrayBodyPtr& a, const std::string& mode) -> py::object {
                Array a_array{a};

                QrMode qrmode{};
                if (mode == "reduced") {
                    qrmode = QrMode::kReduced;
                } else if (mode == "complete") {
                    qrmode = QrMode::kComplete;
                } else if (mode == "r") {
                    qrmode = QrMode::kR;
                } else if (mode == "raw") {
                    qrmode = QrMode::kRaw;
                } else {
                    throw py::value_error{"mode must be 'reduced', 'complete', 'r', or 'raw'"};
                }
                std::tuple<Array, Array> qr = Qr(a_array, qrmode);
                Array& q = std::get<0>(qr);
                Array& r = std::get<1>(qr);
                if (mode == "r") {
                    return py::cast(MoveArrayBody(std::move(r)));
                }
                return py::make_tuple(MoveArrayBody(std::move(q)), MoveArrayBody(std::move(r)));
            },
            "a"_a,
            "mode"_a = "reduced");
    mlinalg.def("cholesky", [](const ArrayBodyPtr& a) { return MoveArrayBody(Cholesky(Array{a})); }, "a"_a);
    mlinalg.def(
            "eigh",
            [](const ArrayBodyPtr& a, const std::string& UPLO) {
                if (UPLO.length() != 1) {
                    throw py::value_error{"UPLO argument must be 'L' or 'U'."};
                }
                std::tuple<Array, Array> wv = Eigh(Array{a}, UPLO.c_str()[0]);
                Array& w = std::get<0>(wv);
                Array& v = std::get<1>(wv);
                return std::make_tuple(MoveArrayBody(std::move(w)), MoveArrayBody(std::move(v)));
            },
            "a"_a,
            "UPLO"_a = "L");
    mlinalg.def(
            "eigvalsh",
            [](const ArrayBodyPtr& a, const std::string& UPLO) {
                if (UPLO.length() != 1) {
                    throw py::value_error{"UPLO argument must be 'L' or 'U'."};
                }
                return MoveArrayBody(Eigvalsh(Array{a}, UPLO.c_str()[0]));
            },
            "a"_a,
            "UPLO"_a = "L");
}

void InitChainerxLogic(pybind11::module& m) {
    // logic routines
    m.def("equal",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Equal(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("not_equal",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(NotEqual(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("greater",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Greater(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("greater_equal",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(GreaterEqual(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("less", [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Less(Array{x1}, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("less_equal",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(LessEqual(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("logical_and",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(LogicalAnd(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("logical_or",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(LogicalOr(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("logical_not", [](const ArrayBodyPtr& x) { return MoveArrayBody(LogicalNot(Array{x})); }, "x"_a);
    m.def("logical_xor",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(LogicalXor(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("all",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(All(Array{a}, Axes{axis}, keepdims)); },
          "a"_a,
          "axis"_a,
          "keepdims"_a = false);
    m.def("all",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(All(Array{a}, ToAxes(axis), keepdims));
          },
          "a"_a,
          "axis"_a = nullptr,
          "keepdims"_a = false);
    m.def("any",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(Any(Array{a}, Axes{axis}, keepdims)); },
          "a"_a,
          "axis"_a,
          "keepdims"_a = false);
    m.def("any",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Any(Array{a}, ToAxes(axis), keepdims));
          },
          "a"_a,
          "axis"_a = nullptr,
          "keepdims"_a = false);
    m.def("isnan", [](const ArrayBodyPtr& x) { return MoveArrayBody(IsNan(Array{x})); }, "x"_a);
    m.def("isinf", [](const ArrayBodyPtr& x) { return MoveArrayBody(IsInf(Array{x})); }, "x"_a);
    m.def("isfinite", [](const ArrayBodyPtr& x) { return MoveArrayBody(IsFinite(Array{x})); }, "x"_a);
}

template <class T1, class T2>
std::vector<ArrayBodyPtr> SwitchBySplitArgs(
        T1& split_sections, T2& split_indices, const ArrayBodyPtr& ary, py::handle indices_or_sections, int8_t axis) {
    // TODO(niboshi): Perhaps we would want more general approach to handle multi-type arguments like indices_or_sections to
    // provide more helpful error message for users.

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
    throw py::type_error{std::string{"indices_or_sections not understood: "} + py::cast<std::string>(py::repr(indices_or_sections))};
}

std::vector<ArrayBodyPtr> SplitByIndicesOrSections(const ArrayBodyPtr& ary, py::handle indices_or_sections, int8_t axis) {
    auto split_sections = [](const ArrayBodyPtr& ary, int64_t sections, int8_t axis) {
        return MoveArrayBodies(Split(Array{ary}, sections, axis));
    };
    auto split_indices = [](const ArrayBodyPtr& ary, const std::vector<int64_t>& indices, int8_t axis) {
        return MoveArrayBodies(Split(Array{ary}, indices, axis));
    };
    return SwitchBySplitArgs(split_sections, split_indices, ary, indices_or_sections, axis);
}

std::vector<ArrayBodyPtr> DSplitByIndicesOrSections(const ArrayBodyPtr& ary, py::handle indices_or_sections) {
    auto split_sections = [](const ArrayBodyPtr& ary, int64_t sections, int8_t /*axis*/) {
        return MoveArrayBodies(DSplit(Array{ary}, sections));
    };
    auto split_indices = [](const ArrayBodyPtr& ary, const std::vector<int64_t>& indices, int8_t /*axis*/) {
        return MoveArrayBodies(DSplit(Array{ary}, indices));
    };
    return SwitchBySplitArgs(split_sections, split_indices, ary, indices_or_sections, 2);
}

std::vector<ArrayBodyPtr> VSplitByIndicesOrSections(const ArrayBodyPtr& ary, py::handle indices_or_sections) {
    auto split_sections = [](const ArrayBodyPtr& ary, int64_t sections, int8_t /*axis*/) {
        return MoveArrayBodies(VSplit(Array{ary}, sections));
    };
    auto split_indices = [](const ArrayBodyPtr& ary, const std::vector<int64_t>& indices, int8_t /*axis*/) {
        return MoveArrayBodies(VSplit(Array{ary}, indices));
    };
    return SwitchBySplitArgs(split_sections, split_indices, ary, indices_or_sections, 0);
}

std::vector<ArrayBodyPtr> HSplitByIndicesOrSections(const ArrayBodyPtr& ary, py::handle indices_or_sections) {
    auto split_sections = [](const ArrayBodyPtr& ary, int64_t sections, int8_t /*axis*/) {
        return MoveArrayBodies(HSplit(Array{ary}, sections));
    };
    auto split_indices = [](const ArrayBodyPtr& ary, const std::vector<int64_t>& indices, int8_t /*axis*/) {
        return MoveArrayBodies(HSplit(Array{ary}, indices));
    };
    return SwitchBySplitArgs(split_sections, split_indices, ary, indices_or_sections, 1);
}

void InitChainerxManipulation(pybind11::module& m) {
    // manipulation routines
    m.def("transpose",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axes) {
              return MoveArrayBody(Transpose(Array{a}, ToAxes(axes)));
          },
          "a"_a,
          "axes"_a = nullptr);
    m.def("transpose",
          [](const ArrayBodyPtr& a, int8_t axes) { return MoveArrayBody(Transpose(Array{a}, {axes})); },
          "a"_a,
          "axes"_a = nullptr);
    m.def("flip",
          [](const ArrayBodyPtr& m, const absl::optional<std::vector<int8_t>>& axes) {
              return MoveArrayBody(Flip(Array{m}, ToAxes(axes)));
          },
          "m"_a,
          "axes"_a = nullptr);
    m.def("flip", [](const ArrayBodyPtr& m, int8_t axes) { return MoveArrayBody(Flip(Array{m}, {axes})); }, "m"_a, "axes"_a = nullptr);
    m.def("fliplr", [](const ArrayBodyPtr& m) { return MoveArrayBody(Fliplr(Array{m})); }, "m"_a);
    m.def("flipud", [](const ArrayBodyPtr& m) { return MoveArrayBody(Flipud(Array{m})); }, "m"_a);
    m.def("ravel", [](const ArrayBodyPtr& a) { return MoveArrayBody(Ravel(Array{a})); }, "a"_a);
    m.def("rollaxis",
          [](const ArrayBodyPtr& a, int8_t axis, int8_t start) { return MoveArrayBody(RollAxis(Array{a}, axis, start)); },
          "a"_a,
          "axis"_a,
          "start"_a = 0);
    m.def("reshape",
          [](const ArrayBodyPtr& a, py::handle newshape) { return MoveArrayBody(Reshape(Array{a}, ToShape(newshape))); },
          "a"_a,
          "newshape"_a);
    m.def("reshape",
          [](const ArrayBodyPtr& a, const std::vector<int64_t>& newshape) {
              return MoveArrayBody(Reshape(Array{a}, {newshape.begin(), newshape.end()}));
          },
          "a"_a,
          "newshape"_a);
    m.def("reshape",
          [](const ArrayBodyPtr& a, py::args args) {
              if (args.size() == 0) {
                  throw ChainerxError{"Reshape is missing shape argument."};
              }
              return MoveArrayBody(Reshape(Array{a}, ToShape(args)));
          },
          "a"_a);
    m.def("squeeze",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(Squeeze(Array{a}, ToAxes(axis)));
          },
          "a"_a,
          "axis"_a = nullptr);
    m.def("squeeze", [](const ArrayBodyPtr& a, int8_t axis) { return MoveArrayBody(Squeeze(Array{a}, Axes{axis})); }, "a"_a, "axis"_a);
    m.def("expand_dims", [](const ArrayBodyPtr& a, int8_t axis) { return MoveArrayBody(ExpandDims(Array{a}, axis)); }, "a"_a, "axis"_a);
    m.def("swapaxes",
          [](const ArrayBodyPtr& a, int8_t axis1, int8_t axis2) { return MoveArrayBody(Swapaxes(Array{a}, axis1, axis2)); },
          "a"_a,
          "axis1"_a,
          "axis2"_a);
    m.def("repeat",
          [](const ArrayBodyPtr& a, int64_t repeats, absl::optional<int8_t> axis) {
              return MoveArrayBody(Repeat(Array{a}, repeats, axis));
          },
          "a"_a,
          "repeats"_a,
          "axis"_a = nullptr);
    m.def("repeat",
          [](const ArrayBodyPtr& a, const std::vector<int64_t>& repeats, absl::optional<int8_t> axis) {
              return MoveArrayBody(Repeat(Array{a}, repeats, axis));
          },
          "a"_a,
          "repeats"_a,
          "axis"_a = nullptr);
    m.def("broadcast_to",
          [](const ArrayBodyPtr& array, py::handle shape) { return MoveArrayBody(Array{array}.BroadcastTo(ToShape(shape))); },
          "array"_a,
          "shape"_a);
    m.def("concatenate",
          [](py::sequence arrays, absl::optional<int8_t> axis) {
              std::vector<Array> xs;
              xs.reserve(arrays.size());
              std::transform(arrays.begin(), arrays.end(), std::back_inserter(xs), [](const auto& item) {
                  return Array{py::cast<ArrayBodyPtr>(item)};
              });
              return MoveArrayBody(Concatenate(xs, axis));
          },
          "arrays"_a,
          "axis"_a = 0);
    m.def("stack",
          [](py::sequence arrays, int8_t axis) {
              std::vector<Array> xs;
              xs.reserve(arrays.size());
              std::transform(arrays.begin(), arrays.end(), std::back_inserter(xs), [](const auto& item) {
                  return Array{py::cast<ArrayBodyPtr>(item)};
              });
              return MoveArrayBody(Stack(xs, axis));
          },
          "arrays"_a,
          "axis"_a = 0);
    m.def("atleast_2d", [](const ArrayBodyPtr& a) { return MoveArrayBody(AtLeast2D(Array{a})); }, "a"_a);
    m.def("atleast_3d", [](const ArrayBodyPtr& a) { return MoveArrayBody(AtLeast3D(Array{a})); }, "a"_a);
    m.def("hstack",
          [](py::sequence arrays) {
              std::vector<Array> xs;
              xs.reserve(arrays.size());
              std::transform(arrays.begin(), arrays.end(), std::back_inserter(xs), [](const auto& item) {
                  return Array{py::cast<ArrayBodyPtr>(item)};
              });
              return MoveArrayBody(HStack(xs));
          },
          "arrays"_a);
    m.def("vstack",
          [](py::sequence arrays) {
              std::vector<Array> xs;
              xs.reserve(arrays.size());
              std::transform(arrays.begin(), arrays.end(), std::back_inserter(xs), [](const auto& item) {
                  return Array{py::cast<ArrayBodyPtr>(item)};
              });
              return MoveArrayBody(VStack(xs));
          },
          "arrays"_a);
    m.def("dstack",
          [](py::sequence arrays) {
              std::vector<Array> xs;
              xs.reserve(arrays.size());
              std::transform(arrays.begin(), arrays.end(), std::back_inserter(xs), [](const auto& item) {
                  return Array{py::cast<ArrayBodyPtr>(item)};
              });
              return MoveArrayBody(DStack(xs));
          },
          "arrays"_a);
    m.def("split", &SplitByIndicesOrSections, "ary"_a, "indices_or_sections"_a, "axis"_a = 0);
    m.def("dsplit", &DSplitByIndicesOrSections, "ary"_a, "indices_or_sections"_a);
    m.def("vsplit", &VSplitByIndicesOrSections, "ary"_a, "indices_or_sections"_a);
    m.def("hsplit", &HSplitByIndicesOrSections, "ary"_a, "indices_or_sections"_a);
    m.def("moveaxis",
          [](const ArrayBodyPtr& a, const std::vector<int8_t>& source, const std::vector<int8_t>& destination) {
              return MoveArrayBody(Moveaxis(Array{a}, Axes{source.begin(), source.end()}, Axes{destination.begin(), destination.end()}));
          },
          "a"_a,
          "source"_a = nullptr,
          "destination"_a = nullptr);
    m.def("moveaxis",
          [](const ArrayBodyPtr& a, py::tuple source, py::tuple destination) {
              return MoveArrayBody(Moveaxis(Array{a}, ToAxes(source), ToAxes(destination)));
          },
          "a"_a,
          "source"_a = nullptr,
          "destination"_a = nullptr);
    m.def("moveaxis",
          [](const ArrayBodyPtr& a, int8_t source, int8_t destination) {
              return MoveArrayBody(Moveaxis(Array{a}, {source}, {destination}));
          },
          "a"_a,
          "source"_a = nullptr,
          "destination"_a = nullptr);
    m.def("copyto",
          [](const ArrayBodyPtr& dst, const ArrayBodyPtr& src, const std::string& casting, Scalar where) {
              CopyTo(Array{dst}, Array{src}, ParseCastingMode(casting), Full({}, where, Dtype::kBool));
          },
          "dst"_a,
          "src"_a,
          "casting"_a = "no",
          "where"_a = true);
    m.def("copyto",
          [](const ArrayBodyPtr& dst, const ArrayBodyPtr& src, const std::string& casting, const ArrayBodyPtr& where) {
              CopyTo(Array{dst}, Array{src}, ParseCastingMode(casting), Array{where});
          },
          "dst"_a,
          "src"_a,
          "casting"_a = "no",
          "where"_a);
}

void InitChainerxActivation(pybind11::module& m) {
    m.def("clipped_relu", [](const ArrayBodyPtr& x, Scalar z) { return MoveArrayBody(ClippedRelu(Array{x}, z)); }, "x"_a, "z"_a = 20.0);
    m.def("crelu", [](const ArrayBodyPtr& x, int8_t axis) { return MoveArrayBody(CRelu(Array{x}, axis)); }, "x"_a, "axis"_a = 1);
    m.def("elu", [](const ArrayBodyPtr& x, double alpha) { return MoveArrayBody(Elu(Array{x}, alpha)); }, "x"_a, "alpha"_a = 1.0);
    m.def("sigmoid", [](const ArrayBodyPtr& x) { return MoveArrayBody(Sigmoid(Array{x})); }, "x"_a);
    m.def("relu", [](const ArrayBodyPtr& x) { return MoveArrayBody(Relu(Array{x})); }, "x"_a);
    m.def("leaky_relu",
          [](const ArrayBodyPtr& x, Scalar slope) { return MoveArrayBody(LeakyRelu(Array{x}, slope)); },
          "x"_a,
          "slope"_a = 0.2);
    m.def("tree_lstm", [](py::args args) {
        std::vector<ArrayBodyPtr> arrays = py::cast<std::vector<ArrayBodyPtr>>(args);
        std::vector<Array> input = ArrayBodiesToArrays(std::move(arrays));
        return ToTuple(TreeLstm(input));
    });
    m.def("slstm", [](const ArrayBodyPtr& c1, const ArrayBodyPtr& c2, const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) {
        return ToTuple(SLstm(Array{c1}, Array{c2}, Array{x1}, Array{x2}));
    });
    m.def("softplus", [](const ArrayBodyPtr& x, double beta) { return MoveArrayBody(Softplus(Array{x}, beta)); }, "x"_a, "beta"_a = 1.0);
}

void InitChainerxArithmetic(pybind11::module& m) {
    // math routines
    m.def("negative", [](const ArrayBodyPtr& x) { return MoveArrayBody(Negative(Array{x})); }, "x"_a);
    m.def("add", [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Array{x1} + Array{x2}); }, "x1"_a, "x2"_a);
    m.def("add", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Add(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("add", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Add(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("subtract", [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Array{x1} - Array{x2}); }, "x1"_a, "x2"_a);
    m.def("subtract", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Subtract(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("subtract", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Subtract(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("multiply", [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Array{x1} * Array{x2}); }, "x1"_a, "x2"_a);
    m.def("multiply", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Multiply(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("multiply", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Multiply(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("divide", [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Array{x1} / Array{x2}); }, "x1"_a, "x2"_a);
    m.def("divide", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Divide(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("divide", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Divide(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("floor_divide",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(FloorDivide(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("floor_divide", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(FloorDivide(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("floor_divide", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(FloorDivide(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("true_divide",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(TrueDivide(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("true_divide", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(TrueDivide(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("true_divide", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(TrueDivide(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("reciprocal", [](const ArrayBodyPtr& x) { return MoveArrayBody(Reciprocal(Array{x})); }, "x"_a);
    m.def("power",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Power(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("power", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Power(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("power", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Power(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("remainder",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Mod(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("remainder", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Mod(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("remainder", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Mod(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.attr("mod") = m.attr("remainder");
    m.def("fmod", [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Fmod(Array{x1}, Array{x2})); }, "x1"_a, "x2"_a);
}

void InitChainerxBinary(pybind11::module& m) {
    m.def("bitwise_and",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(BitwiseAnd(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("bitwise_and", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(BitwiseAnd(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("bitwise_and", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(BitwiseAnd(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("bitwise_or",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(BitwiseOr(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("bitwise_or", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(BitwiseOr(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("bitwise_or", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(BitwiseOr(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("bitwise_xor",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(BitwiseXor(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("bitwise_xor", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(BitwiseXor(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("bitwise_xor", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(BitwiseXor(x1, Array{x2})); }, "x1"_a, "x2"_a);
}

void InitChainerxExpLog(pybind11::module& m) {
    m.def("erf", [](const ArrayBodyPtr& x) { return MoveArrayBody(Erf(Array{x})); }, "x"_a);
    m.def("exp", [](const ArrayBodyPtr& x) { return MoveArrayBody(Exp(Array{x})); }, "x"_a);
    m.def("expm1", [](const ArrayBodyPtr& x) { return MoveArrayBody(Expm1(Array{x})); }, "x"_a);
    m.def("exp2", [](const ArrayBodyPtr& x) { return MoveArrayBody(Exp2(Array{x})); }, "x"_a);
    m.def("log", [](const ArrayBodyPtr& x) { return MoveArrayBody(Log(Array{x})); }, "x"_a);
    m.def("log10", [](const ArrayBodyPtr& x) { return MoveArrayBody(Log10(Array{x})); }, "x"_a);
    m.def("log2", [](const ArrayBodyPtr& x) { return MoveArrayBody(Log2(Array{x})); }, "x"_a);
    m.def("log1p", [](const ArrayBodyPtr& x) { return MoveArrayBody(Log1p(Array{x})); }, "x"_a);
}

void InitChainerxHyperbolic(pybind11::module& m) {
    m.def("sinh", [](const ArrayBodyPtr& x) { return MoveArrayBody(Sinh(Array{x})); }, "x"_a);
    m.def("cosh", [](const ArrayBodyPtr& x) { return MoveArrayBody(Cosh(Array{x})); }, "x"_a);
    m.def("tanh", [](const ArrayBodyPtr& x) { return MoveArrayBody(Tanh(Array{x})); }, "x"_a);
    m.def("arcsinh", [](const ArrayBodyPtr& x) { return MoveArrayBody(Arcsinh(Array{x})); }, "x"_a);
    m.def("arccosh", [](const ArrayBodyPtr& x) { return MoveArrayBody(Arccosh(Array{x})); }, "x"_a);
}

void InitChainerxMisc(pybind11::module& m) {
    m.def("square", [](const ArrayBodyPtr& x) { return MoveArrayBody(Square(Array{x})); }, "x"_a);
    m.def("sqrt", [](const ArrayBodyPtr& x) { return MoveArrayBody(Sqrt(Array{x})); }, "x"_a);
    m.def("abs", [](const ArrayBodyPtr& x) { return MoveArrayBody(Absolute(Array{x})); }, "x"_a);
    m.attr("absolute") = m.attr("abs");
    m.def("fabs", [](const ArrayBodyPtr& x) { return MoveArrayBody(Fabs(Array{x})); }, "x"_a);
    m.def("sign", [](const ArrayBodyPtr& x) { return MoveArrayBody(Sign(Array{x})); }, "x"_a);
    m.def("maximum", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Maximum(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("maximum", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Maximum(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("maximum",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Maximum(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("minimum", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(Minimum(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("minimum", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Minimum(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("minimum",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Minimum(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
}

void InitChainerxReduction(pybind11::module& m) {
    m.def("sum",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(Sum(Array{a}, Axes{axis}, keepdims)); },
          "a"_a,
          "axis"_a,
          "keepdims"_a = false);
    m.def("sum",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Sum(Array{a}, ToAxes(axis), keepdims));
          },
          "a"_a,
          "axis"_a = nullptr,
          "keepdims"_a = false);
    m.def("logsumexp",
          [](const ArrayBodyPtr& x, int8_t axis, bool keepdims) { return MoveArrayBody(LogSumExp(Array{x}, Axes{axis}, keepdims)); },
          "x"_a,
          "axis"_a,
          "keepdims"_a = false);
    m.def("logsumexp",
          [](const ArrayBodyPtr& x, const absl::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(LogSumExp(Array{x}, ToAxes(axis), keepdims));
          },
          "x"_a,
          "axis"_a = nullptr,
          "keepdims"_a = false);
    m.def("log_softmax",
          [](const ArrayBodyPtr& x, int8_t axis) { return MoveArrayBody(LogSoftmax(Array{x}, Axes{axis})); },
          "x"_a,
          "axis"_a);
    m.def("log_softmax",
          [](const ArrayBodyPtr& x, const absl::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(LogSoftmax(Array{x}, ToAxes(axis)));
          },
          "x"_a,
          "axis"_a = nullptr);
    m.def("softmax", [](const ArrayBodyPtr& x, int8_t axis) { return MoveArrayBody(Softmax(Array{x}, Axes{axis})); }, "x"_a, "axis"_a);
    m.def("softmax",
          [](const ArrayBodyPtr& x, const absl::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(Softmax(Array{x}, ToAxes(axis)));
          },
          "x"_a,
          "axis"_a = nullptr);
    m.def("cumsum",
          [](const ArrayBodyPtr& a, absl::optional<int8_t> axis) { return MoveArrayBody(Cumsum(Array{a}, axis)); },
          "a"_a,
          "axis"_a = nullptr);
    m.def("nansum",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(Nansum(Array{a}, Axes{axis}, keepdims)); },
          "a"_a,
          "axis"_a,
          "keepdims"_a = false);
    m.def("nansum",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Nansum(Array{a}, ToAxes(axis), keepdims));
          },
          "a"_a,
          "axis"_a = nullptr,
          "keepdims"_a = false);
}

void InitChainerxRounding(pybind11::module& m) {
    m.def("ceil", [](const ArrayBodyPtr& x) { return MoveArrayBody(Ceil(Array{x})); }, "x"_a);
    m.def("floor", [](const ArrayBodyPtr& x) { return MoveArrayBody(Floor(Array{x})); }, "x"_a);
}

void InitChainerxTrigonometric(pybind11::module& m) {
    m.def("sin", [](const ArrayBodyPtr& x) { return MoveArrayBody(Sin(Array{x})); }, "x"_a);
    m.def("cos", [](const ArrayBodyPtr& x) { return MoveArrayBody(Cos(Array{x})); }, "x"_a);
    m.def("tan", [](const ArrayBodyPtr& x) { return MoveArrayBody(Tan(Array{x})); }, "x"_a);
    m.def("arcsin", [](const ArrayBodyPtr& x) { return MoveArrayBody(Arcsin(Array{x})); }, "x"_a);
    m.def("arccos", [](const ArrayBodyPtr& x) { return MoveArrayBody(Arccos(Array{x})); }, "x"_a);
    m.def("arctan", [](const ArrayBodyPtr& x) { return MoveArrayBody(Arctan(Array{x})); }, "x"_a);
    m.def("arctan2",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(Arctan2(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("left_shift",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(LeftShift(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("left_shift", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(LeftShift(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("left_shift", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(LeftShift(x1, Array{x2})); }, "x1"_a, "x2"_a);
    m.def("right_shift",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(RightShift(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("right_shift", [](const ArrayBodyPtr& x1, Scalar x2) { return MoveArrayBody(RightShift(Array{x1}, x2)); }, "x1"_a, "x2"_a);
    m.def("right_shift", [](Scalar x1, const ArrayBodyPtr& x2) { return MoveArrayBody(RightShift(x1, Array{x2})); }, "x1"_a, "x2"_a);
}

void InitChainerxSorting(pybind11::module& m) {
    // sorting routines
    m.def("argmax",
          [](const ArrayBodyPtr& a, absl::optional<int8_t> axis) { return MoveArrayBody(ArgMax(Array{a}, ToAxes(axis))); },
          "a"_a,
          "axis"_a = nullptr);
    m.def("argmin",
          [](const ArrayBodyPtr& a, absl::optional<int8_t> axis) { return MoveArrayBody(ArgMin(Array{a}, ToAxes(axis))); },
          "a"_a,
          "axis"_a = nullptr);
    m.def("count_nonzero",
          [](const ArrayBodyPtr& a, int8_t axis) { return MoveArrayBody(CountNonzero(Array{a}, Axes{axis})); },
          "a"_a,
          "axis"_a);
    m.def("count_nonzero",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(CountNonzero(Array{a}, ToAxes(axis)));
          },
          "a"_a,
          "axis"_a = nullptr);
    m.def("nanargmax",
          [](const ArrayBodyPtr& a, absl::optional<int8_t> axis) { return MoveArrayBody(NanArgMax(Array{a}, ToAxes(axis))); },
          "a"_a,
          "axis"_a = nullptr);
    m.def("nanargmin",
          [](const ArrayBodyPtr& a, absl::optional<int8_t> axis) { return MoveArrayBody(NanArgMin(Array{a}, ToAxes(axis))); },
          "a"_a,
          "axis"_a = nullptr);
}

void InitChainerxStatistics(pybind11::module& m) {
    // statistics routines
    m.def("amax",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(AMax(Array{a}, Axes{axis}, keepdims)); },
          "a"_a,
          "axis"_a,
          "keepdims"_a = false);
    m.def("amax",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(AMax(Array{a}, ToAxes(axis), keepdims));
          },
          "a"_a,
          "axis"_a = nullptr,
          "keepdims"_a = false);
    m.attr("max") = m.attr("amax");
    m.def("amin",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(AMin(Array{a}, Axes{axis}, keepdims)); },
          "a"_a,
          "axis"_a,
          "keepdims"_a = false);
    m.def("amin",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(AMin(Array{a}, ToAxes(axis), keepdims));
          },
          "a"_a,
          "axis"_a = nullptr,
          "keepdims"_a = false);
    m.attr("min") = m.attr("amin");
    m.def("mean",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(Mean(Array{a}, Axes{axis}, keepdims)); },
          "a"_a,
          "axis"_a,
          "keepdims"_a = false);
    m.def("mean",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Mean(Array{a}, ToAxes(axis), keepdims));
          },
          "a"_a,
          "axis"_a = nullptr,
          "keepdims"_a = false);
    m.def("var",
          [](const ArrayBodyPtr& a, int8_t axis, bool keepdims) { return MoveArrayBody(Var(Array{a}, Axes{axis}, keepdims)); },
          "a"_a,
          "axis"_a,
          "keepdims"_a = false);
    m.def("var",
          [](const ArrayBodyPtr& a, const absl::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Var(Array{a}, ToAxes(axis), keepdims));
          },
          "a"_a,
          "axis"_a = nullptr,
          "keepdims"_a = false);
}

void InitChainerxConnection(pybind11::module& m) {
    // connection routines
    m.def("conv",
          [](const ArrayBodyPtr& x,
             const ArrayBodyPtr& w,
             const absl::optional<ArrayBodyPtr>& b,
             py::handle stride,
             py::handle pad,
             bool cover_all) {
              // Create an Array from x to compute the image dimensions and the expected number of stride and padding elements.
              Array x_array{x};
              int8_t ndim = x_array.ndim() - 2;
              return MoveArrayBody(
                      Conv(x_array,
                           Array{w},
                           b.has_value() ? absl::optional<Array>{Array{*b}} : absl::nullopt,
                           ToStackVector<int64_t>(stride, ndim),
                           ToStackVector<int64_t>(pad, ndim),
                           cover_all));
          },
          "x"_a,
          "w"_a,
          "b"_a = nullptr,
          "stride"_a = 1,
          "pad"_a = 0,
          "cover_all"_a = false);
    m.def("conv_transpose",
          [](const ArrayBodyPtr& x,
             const ArrayBodyPtr& w,
             const absl::optional<ArrayBodyPtr>& b,
             py::handle stride,
             py::handle pad,
             const absl::optional<py::tuple>& outsize) {
              // Create an Array from x to compute the image dimensions and the expected number of stride and padding elements.
              Array x_array{x};
              int8_t ndim = x_array.ndim() - 2;
              return MoveArrayBody(ConvTranspose(
                      x_array,
                      Array{w},
                      b.has_value() ? absl::optional<Array>{Array{*b}} : absl::nullopt,
                      ToStackVector<int64_t>(stride, ndim),
                      ToStackVector<int64_t>(pad, ndim),
                      outsize.has_value() ? absl::optional<Dims>{ToStackVector<int64_t>(*outsize, ndim)} : absl::nullopt));
          },
          "x"_a,
          "w"_a,
          "b"_a = nullptr,
          "stride"_a = 1,
          "pad"_a = 0,
          "outsize"_a = nullptr);
    m.def("linear",
          [](const ArrayBodyPtr& x, const ArrayBodyPtr& w, const absl::optional<ArrayBodyPtr>& b, int8_t n_batch_axes) {
              return MoveArrayBody(
                      Linear(Array{x}, Array{w}, b.has_value() ? absl::optional<Array>{Array{*b}} : absl::nullopt, n_batch_axes));
          },
          "x"_a,
          "w"_a,
          "b"_a = nullptr,
          "n_batch_axes"_a = 1);
    m.def("lstm",
          [](const ArrayBodyPtr& c, const ArrayBodyPtr& x) {
              std::vector<ArrayBodyPtr> out = ToArrayBodyPtr(Lstm(Array{c}, Array{x}));
              py::tuple ret{2};
              ret[0] = out[1];
              ret[1] = out[0];
              return ret;
          },
          py::arg("c"),
          py::arg("x"));
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
             const absl::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(
                      BatchNorm(Array{x}, Array{gamma}, Array{beta}, Array{running_mean}, Array{running_var}, eps, decay, ToAxes(axis)));
          },
          "x"_a,
          "gamma"_a,
          "beta"_a,
          "running_mean"_a,
          "running_var"_a,
          "eps"_a = 2e-5,
          "decay"_a = 0.9,
          "axis"_a = nullptr);
    m.def("fixed_batch_norm",
          [](const ArrayBodyPtr& x,
             const ArrayBodyPtr& gamma,
             const ArrayBodyPtr& beta,
             const ArrayBodyPtr& mean,
             const ArrayBodyPtr& var,
             Scalar eps,
             const absl::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(FixedBatchNorm(Array{x}, Array{gamma}, Array{beta}, Array{mean}, Array{var}, eps, ToAxes(axis)));
          },
          "x"_a,
          "gamma"_a,
          "beta"_a,
          "mean"_a,
          "var"_a,
          "eps"_a = 2e-5,
          "axis"_a = nullptr);
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
          "x"_a,
          "ksize"_a,
          "stride"_a = py::none(),
          "pad"_a = 0,
          "cover_all"_a = false);
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
          "x"_a,
          "ksize"_a,
          "stride"_a = py::none(),
          "pad"_a = 0,
          "pad_mode"_a = "ignore");
}

void InitChainerxLoss(pybind11::module& m) {
    m.def("absolute_error",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(AbsoluteError(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("squared_error",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(SquaredError(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("gaussian_kl_divergence",
          [](const ArrayBodyPtr& mean, const ArrayBodyPtr& ln_var) {
              return MoveArrayBody(GaussianKLDivergence(Array{mean}, Array{ln_var}));
          },
          "mean"_a,
          "ln_var"_a);
    m.def("huber_loss",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2, Scalar delta) {
              return MoveArrayBody(HuberLoss(Array{x1}, Array{x2}, delta));
          },
          "x1"_a,
          "x2"_a,
          "delta"_a);
    m.def("sigmoid_cross_entropy",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(SigmoidCrossEntropy(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("softmax_cross_entropy",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2) { return MoveArrayBody(SoftmaxCrossEntropy(Array{x1}, Array{x2})); },
          "x1"_a,
          "x2"_a);
    m.def("hinge",
          [](const ArrayBodyPtr& x1, const ArrayBodyPtr& x2, double norm) { return MoveArrayBody(Hinge(Array{x1}, Array{x2}, norm)); },
          "x1"_a,
          "x2"_a,
          "norm"_a = 1.0);
}
void InitChainerxRNN(pybind11::module& m) {
    m.def("n_step_lstm",
          [](int64_t n_layers,
             ArrayBodyPtr& hx,
             ArrayBodyPtr& cx,
             std::vector<std::vector<ArrayBodyPtr>> weights,
             std::vector<std::vector<ArrayBodyPtr>> biases,
             std::vector<ArrayBodyPtr> inputs) {
              std::vector<std::vector<Array>> ws;
              std::vector<std::vector<Array>> bs;
              std::vector<Array> xs = ArrayBodiesToArrays(std::move(inputs));
              for (size_t i = 0; i < weights.size(); i++) {
                  std::vector<Array> temp_ws;
                  std::vector<Array> temp_bs;
                  for (size_t j = 0; j < weights[i].size(); j++) {
                      temp_ws.emplace_back(Array{weights[i][j]});
                      temp_bs.emplace_back(Array{biases[i][j]});
                  }
                  ws.emplace_back(temp_ws);
                  bs.emplace_back(temp_bs);
              }
              std::vector<std::vector<Array>> out = NStepLstm(n_layers, Array{hx}, Array{cx}, ws, bs, xs);
              py::tuple ret{3};
              std::vector<ArrayBodyPtr> states = ToArrayBodyPtr(out[0]);
              std::vector<ArrayBodyPtr> ys = ToArrayBodyPtr(out[1]);
              ret[0] = states[0];
              ret[1] = states[1];
              ret[2] = ys;
              return ret;
          });
    m.def("n_step_bilstm",
          [](int64_t n_layers,
             ArrayBodyPtr& hx,
             ArrayBodyPtr& cx,
             std::vector<std::vector<ArrayBodyPtr>> weights,
             std::vector<std::vector<ArrayBodyPtr>> biases,
             std::vector<ArrayBodyPtr> inputs) {
              std::vector<std::vector<Array>> ws;
              std::vector<std::vector<Array>> bs;
              std::vector<Array> xs = ArrayBodiesToArrays(std::move(inputs));
              for (size_t i = 0; i < weights.size(); i++) {
                  std::vector<Array> temp_ws;
                  std::vector<Array> temp_bs;
                  for (size_t j = 0; j < weights[i].size(); j++) {
                      temp_ws.emplace_back(Array{weights[i][j]});
                      temp_bs.emplace_back(Array{biases[i][j]});
                  }
                  ws.emplace_back(temp_ws);
                  bs.emplace_back(temp_bs);
              }
              std::vector<std::vector<Array>> out = NStepBiLstm(n_layers, Array{hx}, Array{cx}, ws, bs, xs);
              py::tuple ret{3};
              std::vector<ArrayBodyPtr> states = ToArrayBodyPtr(out[0]);
              std::vector<ArrayBodyPtr> ys = ToArrayBodyPtr(out[1]);
              ret[0] = states[0];
              ret[1] = states[1];
              ret[2] = ys;
              return ret;
          });
    m.def("n_step_gru",
          [](int64_t n_layers,
             ArrayBodyPtr& hx,
             std::vector<std::vector<ArrayBodyPtr>> weights,
             std::vector<std::vector<ArrayBodyPtr>> biases,
             std::vector<ArrayBodyPtr> inputs) {
              std::vector<std::vector<Array>> ws;
              std::vector<std::vector<Array>> bs;
              std::vector<Array> xs = ArrayBodiesToArrays(std::move(inputs));
              for (size_t i = 0; i < weights.size(); i++) {
                  std::vector<Array> temp_ws;
                  std::vector<Array> temp_bs;
                  for (size_t j = 0; j < weights[i].size(); j++) {
                      temp_ws.emplace_back(Array{weights[i][j]});
                      temp_bs.emplace_back(Array{biases[i][j]});
                  }
                  ws.emplace_back(temp_ws);
                  bs.emplace_back(temp_bs);
              }
              std::vector<std::vector<Array>> out = NStepGru(n_layers, Array{hx}, ws, bs, xs);
              py::tuple ret{2};
              std::vector<ArrayBodyPtr> states = ToArrayBodyPtr(out[0]);
              std::vector<ArrayBodyPtr> ys = ToArrayBodyPtr(out[1]);
              ret[0] = states[0];
              ret[1] = ys;
              return ret;
          });
    m.def("n_step_bigru",
          [](int64_t n_layers,
             ArrayBodyPtr& hx,
             std::vector<std::vector<ArrayBodyPtr>> weights,
             std::vector<std::vector<ArrayBodyPtr>> biases,
             std::vector<ArrayBodyPtr> inputs) {
              std::vector<std::vector<Array>> ws;
              std::vector<std::vector<Array>> bs;
              std::vector<Array> xs = ArrayBodiesToArrays(std::move(inputs));
              for (size_t i = 0; i < weights.size(); i++) {
                  std::vector<Array> temp_ws;
                  std::vector<Array> temp_bs;
                  for (size_t j = 0; j < weights[i].size(); j++) {
                      temp_ws.emplace_back(Array{weights[i][j]});
                      temp_bs.emplace_back(Array{biases[i][j]});
                  }
                  ws.emplace_back(temp_ws);
                  bs.emplace_back(temp_bs);
              }
              std::vector<std::vector<Array>> out = NStepBiGru(n_layers, Array{hx}, ws, bs, xs);
              py::tuple ret{2};
              std::vector<ArrayBodyPtr> states = ToArrayBodyPtr(out[0]);
              std::vector<ArrayBodyPtr> ys = ToArrayBodyPtr(out[1]);
              ret[0] = states[0];
              ret[1] = ys;
              return ret;
          });
    m.def("n_step_rnn",
          [](int64_t n_layers,
             ArrayBodyPtr& hx,
             std::vector<std::vector<ArrayBodyPtr>> weights,
             std::vector<std::vector<ArrayBodyPtr>> biases,
             std::vector<ArrayBodyPtr> inputs,
             std::string activation) {
              std::vector<std::vector<Array>> ws;
              std::vector<std::vector<Array>> bs;
              std::vector<Array> xs = ArrayBodiesToArrays(std::move(inputs));
              for (size_t i = 0; i < weights.size(); i++) {
                  std::vector<Array> temp_ws;
                  std::vector<Array> temp_bs;
                  for (size_t j = 0; j < weights[i].size(); j++) {
                      temp_ws.emplace_back(Array{weights[i][j]});
                      temp_bs.emplace_back(Array{biases[i][j]});
                  }
                  ws.emplace_back(temp_ws);
                  bs.emplace_back(temp_bs);
              }
              std::vector<std::vector<Array>> out = NStepRnn(n_layers, Array{hx}, ws, bs, xs, activation);
              py::tuple ret{2};
              std::vector<ArrayBodyPtr> states = ToArrayBodyPtr(out[0]);
              std::vector<ArrayBodyPtr> ys = ToArrayBodyPtr(out[1]);
              ret[0] = states[0];
              ret[1] = ys;
              return ret;
          });
    m.def("n_step_birnn",
          [](int64_t n_layers,
             ArrayBodyPtr& hx,
             std::vector<std::vector<ArrayBodyPtr>> weights,
             std::vector<std::vector<ArrayBodyPtr>> biases,
             std::vector<ArrayBodyPtr> inputs,
             std::string activation) {
              std::vector<std::vector<Array>> ws;
              std::vector<std::vector<Array>> bs;
              std::vector<Array> xs = ArrayBodiesToArrays(std::move(inputs));
              for (size_t i = 0; i < weights.size(); i++) {
                  std::vector<Array> temp_ws;
                  std::vector<Array> temp_bs;
                  for (size_t j = 0; j < weights[i].size(); j++) {
                      temp_ws.emplace_back(Array{weights[i][j]});
                      temp_bs.emplace_back(Array{biases[i][j]});
                  }
                  ws.emplace_back(temp_ws);
                  bs.emplace_back(temp_bs);
              }
              std::vector<std::vector<Array>> out = NStepBiRnn(n_layers, Array{hx}, ws, bs, xs, activation);
              py::tuple ret{2};
              std::vector<ArrayBodyPtr> states = ToArrayBodyPtr(out[0]);
              std::vector<ArrayBodyPtr> ys = ToArrayBodyPtr(out[1]);
              ret[0] = states[0];
              ret[1] = ys;
              return ret;
          });
}

}  // namespace

void InitChainerxRoutines(pybind11::module& m) {
    InitChainerxCreation(m);
    InitChainerxEvaluation(m);
    InitChainerxIndexing(m);
    InitChainerxLinalg(m);
    InitChainerxLogic(m);
    InitChainerxLoss(m);
    InitChainerxManipulation(m);
    InitChainerxActivation(m);
    InitChainerxArithmetic(m);
    InitChainerxBinary(m);
    InitChainerxExpLog(m);
    InitChainerxHyperbolic(m);
    InitChainerxMisc(m);
    InitChainerxReduction(m);
    InitChainerxRounding(m);
    InitChainerxTrigonometric(m);
    InitChainerxSorting(m);
    InitChainerxStatistics(m);
    InitChainerxConnection(m);
    InitChainerxNormalization(m);
    InitChainerxPooling(m);
    InitChainerxRNN(m);
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
