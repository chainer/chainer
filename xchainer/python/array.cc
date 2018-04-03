#include "xchainer/python/array.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/backward.h"
#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/slice.h"

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

Dtype NumpyDtypeToDtype(const py::dtype& npdtype) {
    switch (npdtype.kind()) {
        case 'b':
            return Dtype::kBool;
        case 'i':
            switch (npdtype.itemsize()) {
                case 1:
                    return Dtype::kInt8;
                case 2:
                    return Dtype::kInt16;
                case 4:
                    return Dtype::kInt32;
                case 8:
                    return Dtype::kInt64;
                default:
                    break;
            }
            break;
        case 'u':
            switch (npdtype.itemsize()) {
                case 1:
                    return Dtype::kUInt8;
                default:
                    break;
            }
            break;
        case 'f':
            switch (npdtype.itemsize()) {
                case 4:
                    return Dtype::kFloat32;
                case 8:
                    return Dtype::kFloat64;
                default:
                    break;
            }
            break;
        default:
            break;
    }
    throw DtypeError("unsupported NumPy dtype");
}

py::buffer_info MakeBufferFromArray(ArrayBody& self) {
    // Used as a temporary accessor
    Array array{ArrayBodyPtr(&self, [](ArrayBody* ptr) {
        (void)ptr;  // unused
    })};

    return py::buffer_info(
            array.data().get(),
            array.element_bytes(),
            std::string(1, GetCharCode(array.dtype())),
            array.ndim(),
            array.shape(),
            array.strides());
}

}  // namespace

ArrayBodyPtr MakeArray(const py::tuple& shape_tup, Dtype dtype, const py::list& list, Device& device) {
    Shape shape = ToShape(shape_tup);
    auto total_size = shape.GetTotalSize();
    auto bytes = GetElementSize(dtype) * total_size;
    if (static_cast<size_t>(total_size) != list.size()) {
        throw DimensionError("Invalid data length");
    }

    // Allocate a buffer and copy data
    std::shared_ptr<void> ptr = std::make_unique<uint8_t[]>(bytes);
    VisitDtype(dtype, [&](auto pt) {
        using T = typename decltype(pt)::type;
        std::transform(list.begin(), list.end(), static_cast<T*>(ptr.get()), [](auto& item) { return py::cast<T>(item); });
    });

    return Array::FromBuffer(shape, dtype, ptr, device).move_body();
}

ArrayBodyPtr MakeArray(py::array array, Device& device) {
    Dtype dtype = NumpyDtypeToDtype(array.dtype());
    const py::buffer_info& info = array.request();
    Shape shape{info.shape};
    Strides strides{info.strides};

    // data holds the copy of py::array which in turn references the NumPy array and the buffer is therefore not released
    void* underlying_data = array.mutable_data();
    std::shared_ptr<void> data{std::make_shared<py::array>(std::move(array)), underlying_data};
    return xchainer::internal::FromBuffer(shape, dtype, data, strides, device).move_body();
}

void InitXchainerArray(pybind11::module& m) {
    py::class_<ArrayBody, ArrayBodyPtr> c{m, "Array", py::buffer_protocol()};
    c.def(py::init([](const py::tuple& shape, Dtype dtype, const py::list& list, const nonstd::optional<std::string>& device_id) {
              return MakeArray(shape, dtype, list, GetDevice(device_id));
          }),
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("data"),
          py::arg("device") = nullptr);
    c.def(py::init([](const py::array& array, const nonstd::optional<std::string>& device_id) {
              return MakeArray(array, GetDevice(device_id));
          }),
          py::arg("data"),
          py::arg("device") = nullptr);
    c.def_buffer(&MakeBufferFromArray);
    c.def("__bool__", [](const ArrayBodyPtr& self) -> bool { return static_cast<bool>(AsScalar(Array{self})); });
    c.def("__int__", [](const ArrayBodyPtr& self) -> int64_t { return static_cast<int64_t>(AsScalar(Array{self})); });
    c.def("__float__", [](const ArrayBodyPtr& self) -> double { return static_cast<double>(AsScalar(Array{self})); });
    c.def("view", [](const ArrayBodyPtr& self) {
        // Duplicate the array body
        return std::make_shared<ArrayBody>(*self);
    });
    c.def("__repr__", [](const ArrayBodyPtr& self) { return Array{self}.ToString(); });
    c.def("to_device", [](const ArrayBodyPtr& self, Device& device) { return Array{self}.ToDevice(device).move_body(); });
    c.def("to_device", [](const ArrayBodyPtr& self, const std::string& device_name) {
        Device& device = GetDefaultContext().GetDevice({device_name});
        return Array{self}.ToDevice(device).move_body();
    });
    c.def("to_device", [](const ArrayBodyPtr& self, const std::string& backend_name, int index) {
        Device& device = GetDefaultContext().GetDevice({backend_name, index});
        return Array{self}.ToDevice(device).move_body();
    });
    c.def("as_constant",
          [](const ArrayBodyPtr& self, bool copy) { return Array{self}.AsConstant(copy ? CopyKind::kCopy : CopyKind::kView).move_body(); },
          py::arg("copy") = false);
    c.def("as_constant",
          [](const ArrayBodyPtr& self, const std::vector<GraphId>& graph_ids, bool copy) {
              return Array{self}.AsConstant(graph_ids, copy ? CopyKind::kCopy : CopyKind::kView).move_body();
          },
          py::arg().noconvert(),
          py::arg("copy") = false);
    c.def("copy", [](const ArrayBodyPtr& self) { return Array{self}.Copy().move_body(); });
    c.def("__getitem__", [](const ArrayBodyPtr& self, py::handle handle) {
        return Array{self}.At(python::internal::MakeArrayIndices(handle)).move_body();
    });
    c.def("transpose", [](const ArrayBodyPtr& self) { return Array{self}.Transpose().move_body(); });
    c.def("reshape", [](const ArrayBodyPtr& self, py::tuple shape) { return Array{self}.Reshape(ToShape(shape)).move_body(); });
    c.def("reshape", [](const ArrayBodyPtr& self, const std::vector<int64_t>& shape) {
        return Array{self}.Reshape({shape.begin(), shape.end()}).move_body();
    });
    c.def("reshape", [](const ArrayBodyPtr& self, py::args args) {
        auto shape = py::cast<std::vector<int64_t>>(args);
        return Array{self}.Reshape({shape.begin(), shape.end()}).move_body();
    });
    c.def("squeeze",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis) { return Array{self}.Squeeze(axis).move_body(); },
          py::arg("axis") = nullptr);
    c.def("squeeze",
          [](const ArrayBodyPtr& self, int8_t axis) { return Array{self}.Squeeze(std::vector<int8_t>{axis}).move_body(); },
          py::arg("axis"));
    c.def("__eq__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} == Array{rhs}).move_body(); });
    c.def("__iadd__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} += Array{rhs}).move_body(); });
    c.def("__imul__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} *= Array{rhs}).move_body(); });
    c.def("__add__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} + Array{rhs}).move_body(); });
    c.def("__mul__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} * Array{rhs}).move_body(); });
    c.def("__mul__", [](const ArrayBodyPtr& self, Scalar rhs) { return (Array{self} * rhs).move_body(); });
    c.def("__rmul__", [](const ArrayBodyPtr& self, Scalar lhs) { return (lhs * Array{self}).move_body(); });
    c.def("sum",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) {
              return Array{self}.Sum(std::vector<int8_t>{axis}, keepdims).move_body();
          },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("sum",
          [](const ArrayBodyPtr& self, nonstd::optional<std::vector<int8_t>> axis, bool keepdims) {
              return Array{self}.Sum(axis, keepdims).move_body();
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("dot", [](const ArrayBodyPtr& self, const ArrayBodyPtr& b) { return Array{self}.Dot(Array{b}).move_body(); }, py::arg("b"));

    c.def("require_grad",
          [](const ArrayBodyPtr& self, const GraphId& graph_id) { return Array{self}.RequireGrad(graph_id).move_body(); },
          py::arg("graph_id") = kDefaultGraphId);
    c.def("is_grad_required",
          [](const ArrayBodyPtr& self, const GraphId& graph_id) { return Array{self}.IsGradRequired(graph_id); },
          py::arg("graph_id") = kDefaultGraphId);
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
    c.def_property_readonly("element_bytes", [](const ArrayBodyPtr& self) { return Array{self}.element_bytes(); });
    c.def_property_readonly("is_contiguous", [](const ArrayBodyPtr& self) { return Array{self}.IsContiguous(); });
    c.def_property_readonly("ndim", [](const ArrayBodyPtr& self) { return Array{self}.ndim(); });
    c.def_property_readonly("offset", [](const ArrayBodyPtr& self) { return Array{self}.offset(); });
    c.def_property_readonly("shape", [](const ArrayBodyPtr& self) { return ToTuple(Array{self}.shape()); });
    c.def_property_readonly("strides", [](const ArrayBodyPtr& self) { return ToTuple(Array{self}.strides()); });
    c.def_property_readonly("total_bytes", [](const ArrayBodyPtr& self) { return Array{self}.GetTotalBytes(); });
    c.def_property_readonly("total_size", [](const ArrayBodyPtr& self) { return Array{self}.GetTotalSize(); });
    c.def_property_readonly("T", [](const ArrayBodyPtr& self) { return Array{self}.Transpose().move_body(); });
    c.def_property_readonly(
            "_debug_data_memory_address",  // These methods starting with `_debug_` are stubs for testing
            [](const ArrayBodyPtr& self) -> intptr_t {
                const void* ptr = Array{self}.data().get();
                return reinterpret_cast<intptr_t>(ptr);  // NOLINT: reinterpret_cast
            });
    c.def_property_readonly("_debug_flat_data", [](const ArrayBodyPtr& self) {
        py::list list;
        Array array{self};
        array.device().Synchronize();

        // Copy data into the list
        VisitDtype(array.dtype(), [&array, &list](auto pt) {
            using T = typename decltype(pt)::type;
            IndexableArray<const T> iarray{array};
            Indexer<> indexer{array.shape()};

            for (int64_t i = 0; i < indexer.total_size(); ++i) {
                indexer.Set(i);
                list.append(iarray[indexer]);
            }
        });

        return list;
    });
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
