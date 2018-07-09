#include "xchainer/python/array.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/operators.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/axes.h"
#include "xchainer/backend_util.h"
#include "xchainer/backward.h"
#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/native_backend.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/indexing.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/routines/sorting.h"
#include "xchainer/shape.h"
#include "xchainer/slice.h"
#include "xchainer/strides.h"

#include "xchainer/python/array_index.h"
#include "xchainer/python/axes.h"
#include "xchainer/python/common.h"
#include "xchainer/python/device.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/shape.h"
#include "xchainer/python/strides.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

ArrayBodyPtr MakeArrayFromNumpyArray(py::array array, Device& device) {
    Shape shape{array.shape(), array.shape() + array.ndim()};
    Dtype dtype = internal::GetDtypeFromNumpyDtype(array.dtype());
    Strides strides{array.strides(), array.strides() + array.ndim()};

    // Copy to a newly allocated data
    std::tuple<int64_t, int64_t> range = GetDataRange(shape, strides, array.itemsize());
    auto bytesize = static_cast<size_t>(std::get<1>(range) - std::get<0>(range));
    std::shared_ptr<void> data = std::make_unique<uint8_t[]>(bytesize);
    {
        py::buffer_info info = array.request();
        std::memcpy(data.get(), info.ptr, bytesize);
    }

    // Create and return the array
    return xchainer::internal::FromHostData(shape, dtype, data, strides, device).move_body();
}

namespace {

ArrayBodyPtr MakeArray(const py::tuple& shape_tup, Dtype dtype, const py::list& list, Device& device) {
    Shape shape = ToShape(shape_tup);
    auto total_size = shape.GetTotalSize();
    auto bytes = GetItemSize(dtype) * total_size;
    if (static_cast<size_t>(total_size) != list.size()) {
        throw DimensionError{"Invalid data length"};
    }

    // Allocate a buffer and copy data
    std::shared_ptr<void> ptr = std::make_unique<uint8_t[]>(bytes);
    VisitDtype(dtype, [&](auto pt) {
        using T = typename decltype(pt)::type;
        std::transform(list.begin(), list.end(), static_cast<T*>(ptr.get()), [](auto& item) { return py::cast<T>(item); });
    });

    return xchainer::internal::FromContiguousHostData(shape, dtype, ptr, device).move_body();
}

py::array MakeNumpyArrayFromArray(const ArrayBodyPtr& self) {
    Array array = Array{self}.ToNative();
    return py::array{py::buffer_info{xchainer::internal::GetRawOffsetData<void>(array),
                                     array.item_size(),
                                     std::string(1, GetCharCode(array.dtype())),
                                     array.ndim(),
                                     array.shape(),
                                     array.strides()}};
}

}  // namespace

ArrayBodyPtr MakeArray(py::handle object, py::handle dtype, bool copy, py::handle device) {
    Device& dev = internal::GetDevice(device);

    // object is xchainer.ndarray
    if (py::isinstance<ArrayBody>(object)) {
        Array a = Array{py::cast<ArrayBodyPtr>(object)};
        Dtype dtype_ = dtype.is_none() ? a.dtype() : internal::GetDtype(dtype);

        if (!copy && a.dtype() == dtype_ && &a.device() == &dev) {
            return a.move_body();
        }
        // Note that the graph is connected.
        if (&a.device() != &dev) {
            return a.ToDevice(dev).AsType(dtype_, false).move_body();
        }
        if (a.dtype() != dtype_) {
            return a.AsType(dtype_, true).move_body();
        }
        return a.Copy().move_body();
    }

    // Convert object to NumPy array using numpy.array()
    // TODO(sonots): Remove dependency on numpy
    py::object array_func = py::module::import("numpy").attr("array");
    py::object dtype_name = py::none();
    if (!dtype.is_none()) {
        dtype_name = py::str{GetDtypeName(internal::GetDtype(dtype))};
    }
    py::array np_array = array_func(object, py::arg("copy") = copy, py::arg("dtype") = dtype_name);

    // Convert NumPy array to Xchainer array
    return MakeArrayFromNumpyArray(np_array, dev);
}

void InitXchainerArray(pybind11::module& m) {
    py::class_<ArrayBody, ArrayBodyPtr> c{m, "ndarray", py::buffer_protocol()};
    // TODO(hvy): Remove list accepting bindings and replace calls with xchainer.array.
    // For multidimensional arrays, nested lists should be passed to xchainer.array.
    c.def(py::init([](const py::tuple& shape, py::handle dtype, const py::list& list, py::handle device) {
              return MakeArray(shape, internal::GetDtype(dtype), list, internal::GetDevice(device));
          }),
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("data") = nullptr,
          py::arg("device") = nullptr);
    // TODO(hvy): Support all arguments in the constructor of numpy.ndarray.
    c.def(py::init([](const py::tuple& shape, py::handle dtype, py::handle device) {
              return Empty(ToShape(shape), internal::GetDtype(dtype), internal::GetDevice(device)).move_body();
          }),
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    m.def("tonumpy", &MakeNumpyArrayFromArray);
    c.def("__bool__", [](const ArrayBodyPtr& self) -> bool { return static_cast<bool>(AsScalar(Array{self})); });
    c.def("__int__", [](const ArrayBodyPtr& self) -> int64_t { return static_cast<int64_t>(AsScalar(Array{self})); });
    c.def("__float__", [](const ArrayBodyPtr& self) -> double { return static_cast<double>(AsScalar(Array{self})); });
    c.def("view", [](const ArrayBodyPtr& self) { return Array{self}.MakeView().move_body(); });
    c.def("__repr__", [](const ArrayBodyPtr& self) { return Array{self}.ToString(); });
    c.def("to_device",
          [](const ArrayBodyPtr& self, py::handle device) { return Array{self}.ToDevice(internal::GetDevice(device)).move_body(); });
    c.def("to_device", [](const ArrayBodyPtr& self, const std::string& backend_name, int index) {
        Device& device = GetDefaultContext().GetDevice({backend_name, index});
        return Array{self}.ToDevice(device).move_body();
    });
    c.def("as_grad_stopped",
          [](const ArrayBodyPtr& self, bool copy) {
              return Array{self}.AsGradStopped(copy ? CopyKind::kCopy : CopyKind::kView).move_body();
          },
          py::arg("copy") = false);
    c.def("as_grad_stopped",
          [](const ArrayBodyPtr& self, const std::vector<GraphId>& graph_ids, bool copy) {
              return Array{self}.AsGradStopped(graph_ids, copy ? CopyKind::kCopy : CopyKind::kView).move_body();
          },
          py::arg().noconvert(),
          py::arg("copy") = false);
    c.def("astype",
          [](const ArrayBodyPtr& self, py::handle dtype, bool copy) {
              return Array{self}.AsType(internal::GetDtype(dtype), copy).move_body();
          },
          py::arg("dtype"),
          py::arg("copy") = true);
    c.def("copy", [](const ArrayBodyPtr& self) { return Array{self}.Copy().move_body(); });
    c.def("__getitem__", [](const ArrayBodyPtr& self, py::handle handle) {
        return Array{self}.At(python::internal::MakeArrayIndices(handle)).move_body();
    });
    c.def("take",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& indices, const nonstd::optional<int8_t>& axis) {
              if (!axis.has_value()) {
                  throw NotImplementedError{"axis=None is not yet supported for xchainer.ndarray.take."};
              }
              return Array{self}.Take(Array{indices}, axis.value()).move_body();
          },
          py::arg("indices"),
          py::arg("axis") = nullptr);
    c.def("transpose",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axes) {
              return Array{self}.Transpose(ToAxes(axes)).move_body();
          },
          py::arg("axes") = nullptr);
    c.def("transpose", [](const ArrayBodyPtr& self, py::args args) { return Array{self}.Transpose(ToAxes(args)).move_body(); });
    c.def("reshape", [](const ArrayBodyPtr& self, py::tuple shape) { return Array{self}.Reshape(ToShape(shape)).move_body(); });
    c.def("reshape", [](const ArrayBodyPtr& self, const std::vector<int64_t>& shape) {
        return Array{self}.Reshape({shape.begin(), shape.end()}).move_body();
    });
    c.def("reshape", [](const ArrayBodyPtr& self, py::args args) {
        if (args.size() == 0) {
            throw XchainerError("Reshape takes exactly 1 argument (0 given).");
        }
        return Array{self}.Reshape(ToShape(args)).move_body();
    });
    c.def("squeeze",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis) {
              return Array{self}.Squeeze(ToAxes(axis)).move_body();
          },
          py::arg("axis") = nullptr);
    c.def("squeeze", [](const ArrayBodyPtr& self, int8_t axis) { return Array{self}.Squeeze(Axes{axis}).move_body(); }, py::arg("axis"));
    c.def("__eq__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} == Array{rhs}).move_body(); });
    c.def("__neg__", [](const ArrayBodyPtr& self) { return (-Array{self}).move_body(); });
    c.def("__iadd__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} += Array{rhs}).move_body(); });
    c.def("__iadd__", [](const ArrayBodyPtr& self, Scalar rhs) { return (Array{self} += rhs).move_body(); });
    c.def("__isub__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} -= Array{rhs}).move_body(); });
    c.def("__isub__", [](const ArrayBodyPtr& self, Scalar rhs) { return (Array{self} -= rhs).move_body(); });
    c.def("__imul__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} *= Array{rhs}).move_body(); });
    c.def("__imul__", [](const ArrayBodyPtr& self, Scalar rhs) { return (Array{self} *= rhs).move_body(); });
    c.def("__itruediv__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} /= Array{rhs}).move_body(); });
    c.def("__itruediv__", [](const ArrayBodyPtr& self, Scalar rhs) { return (Array{self} /= rhs).move_body(); });
    c.def("__add__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} + Array{rhs}).move_body(); });
    c.def("__add__", [](const ArrayBodyPtr& self, Scalar rhs) { return (Array{self} + rhs).move_body(); });
    c.def("__radd__", [](const ArrayBodyPtr& self, Scalar lhs) { return (lhs + Array{self}).move_body(); });
    c.def("__sub__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} - Array{rhs}).move_body(); });
    c.def("__sub__", [](const ArrayBodyPtr& self, Scalar rhs) { return (Array{self} - rhs).move_body(); });
    c.def("__rsub__", [](const ArrayBodyPtr& self, Scalar lhs) { return (lhs - Array{self}).move_body(); });
    c.def("__mul__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} * Array{rhs}).move_body(); });
    c.def("__mul__", [](const ArrayBodyPtr& self, Scalar rhs) { return (Array{self} * rhs).move_body(); });
    c.def("__rmul__", [](const ArrayBodyPtr& self, Scalar lhs) { return (lhs * Array{self}).move_body(); });
    c.def("__truediv__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} / Array{rhs}).move_body(); });
    c.def("__truediv__", [](const ArrayBodyPtr& self, Scalar rhs) { return (Array{self} / rhs).move_body(); });
    c.def("sum",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) { return Array{self}.Sum(Axes{axis}, keepdims).move_body(); },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("sum",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return Array{self}.Sum(ToAxes(axis), keepdims).move_body();
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("max",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) { return Array{self}.Max(Axes{axis}, keepdims).move_body(); },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("max",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return Array{self}.Max(ToAxes(axis), keepdims).move_body();
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("argmax",
          [](const ArrayBodyPtr& self, const nonstd::optional<int8_t>& axis) { return ArgMax(Array{self}, ToAxes(axis)).move_body(); },
          py::arg("axis") = nullptr);
    c.def("dot", [](const ArrayBodyPtr& self, const ArrayBodyPtr& b) { return Array{self}.Dot(Array{b}).move_body(); }, py::arg("b"));
    c.def("fill",
          [](const ArrayBodyPtr& self, Scalar value) {
              Array{self}.Fill(value);
              return;
          },
          py::arg("value"));

    c.def("require_grad",
          [](const ArrayBodyPtr& self, const GraphId& graph_id) { return Array{self}.RequireGrad(graph_id).move_body(); },
          py::arg("graph_id") = kDefaultGraphId);
    c.def("is_grad_required",
          [](const ArrayBodyPtr& self, const GraphId& graph_id) { return Array{self}.IsGradRequired(graph_id); },
          py::arg("graph_id") = kDefaultGraphId);
    c.def("is_grad_required",
          [](const ArrayBodyPtr& self, AnyGraph any_graph) { return Array{self}.IsGradRequired(any_graph); },
          py::arg("graph_id"));
    c.def("get_grad",
          [](const ArrayBodyPtr& self, const GraphId& graph_id) -> ConstArrayBodyPtr {
              const nonstd::optional<Array>& grad = Array{self}.GetGrad(graph_id);
              if (!grad.has_value()) {
                  return nullptr;
              }
              return grad->body();
          },
          py::arg("graph_id") = kDefaultGraphId);
    c.def("set_grad",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& grad, const GraphId& graph_id) {
              auto array = Array{self};
              if (grad) {
                  array.SetGrad(Array{grad}, graph_id);
              } else {
                  array.ClearGrad(graph_id);
              }
          },
          py::arg("grad"),
          py::arg("graph_id") = kDefaultGraphId);
    c.def("backward",
          [](const ArrayBodyPtr& self, const GraphId& graph_id, bool enable_double_backprop) {
              Array array{self};
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward(array, graph_id, double_backprop);
          },
          py::arg("graph_id") = kDefaultGraphId,
          py::arg("enable_double_backprop") = false);
    c.def("_debug_dump_computational_graph",
          [](const ArrayBodyPtr& self, const GraphId& graph_id) {
              Array array{self};
              DebugDumpComputationalGraph(std::cout, array, graph_id);
          },
          py::arg("graph_id") = kDefaultGraphId);
    c.def_property(
            "grad",
            [](const ArrayBodyPtr& self) -> ConstArrayBodyPtr {
                const nonstd::optional<Array>& grad = Array{self}.GetGrad(kDefaultGraphId);
                if (!grad.has_value()) {
                    return nullptr;
                }
                return grad->body();
            },
            [](const ArrayBodyPtr& self, const ArrayBodyPtr& grad) {
                auto array = Array{self};
                if (grad) {
                    array.SetGrad(Array{grad}, kDefaultGraphId);
                } else {
                    array.ClearGrad(kDefaultGraphId);
                }
            });
    c.def("cleargrad",
          [](const ArrayBodyPtr& self, const GraphId& graph_id) { Array{self}.ClearGrad(graph_id); },
          py::arg("graph_id") = kDefaultGraphId);
    c.def_property_readonly(
            "device", [](const ArrayBodyPtr& self) -> Device& { return Array{self}.device(); }, py::return_value_policy::reference);
    c.def_property_readonly("dtype", [](const ArrayBodyPtr& self) { return Array{self}.dtype(); });
    c.def_property_readonly("itemsize", [](const ArrayBodyPtr& self) { return Array{self}.item_size(); });
    c.def_property_readonly("is_contiguous", [](const ArrayBodyPtr& self) { return Array{self}.IsContiguous(); });
    c.def_property_readonly("ndim", [](const ArrayBodyPtr& self) { return Array{self}.ndim(); });
    c.def_property_readonly("offset", [](const ArrayBodyPtr& self) { return Array{self}.offset(); });
    c.def_property_readonly("shape", [](const ArrayBodyPtr& self) { return ToTuple(Array{self}.shape()); });
    c.def_property_readonly("strides", [](const ArrayBodyPtr& self) { return ToTuple(Array{self}.strides()); });
    c.def_property_readonly("nbytes", [](const ArrayBodyPtr& self) { return Array{self}.GetNBytes(); });
    c.def_property_readonly("size", [](const ArrayBodyPtr& self) { return Array{self}.GetTotalSize(); });
    c.def_property_readonly("T", [](const ArrayBodyPtr& self) { return Array{self}.Transpose().move_body(); });
    c.def_property_readonly(
            "_debug_data_memory_address",  // These methods starting with `_debug_` are stubs for testing
            [](const ArrayBodyPtr& self) -> intptr_t {
                const void* ptr = Array{self}.data().get();
                return reinterpret_cast<intptr_t>(ptr);  // NOLINT: reinterpret_cast
            });
    c.def_property_readonly("_debug_flat_data", [](const ArrayBodyPtr& self) {
        py::list list;
        Array array = Array{self}.ToNative();

        // Copy data into the list
        VisitDtype(array.dtype(), [&array, &list](auto pt) {
            using T = typename decltype(pt)::type;
            IndexableArray<const T> iarray{array};
            Indexer<> indexer{array.shape()};

            for (auto it = indexer.It(0); it; ++it) {
                list.append(iarray[it]);
            }
        });

        return list;
    });
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
