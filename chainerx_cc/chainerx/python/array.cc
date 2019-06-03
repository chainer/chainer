#include "chainerx/python/array.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <pybind11/operators.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/axes.h"
#include "chainerx/backend_util.h"
#include "chainerx/backward.h"
#include "chainerx/constant.h"
#include "chainerx/context.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/native/data_type.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/indexing.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/sorting.h"
#include "chainerx/shape.h"
#include "chainerx/slice.h"
#include "chainerx/strides.h"

#include "chainerx/python/array_index.h"
#include "chainerx/python/axes.h"
#include "chainerx/python/common.h"
#include "chainerx/python/device.h"
#include "chainerx/python/dtype.h"
#include "chainerx/python/py_cached_objects.h"
#include "chainerx/python/shape.h"
#include "chainerx/python/strides.h"

namespace chainerx {
namespace python {
namespace python_internal {
namespace {

using internal::MoveArrayBody;

}  // namespace

namespace py = pybind11;

ArrayBodyPtr MakeArrayFromNumpyArray(py::array array, Device& device) {
    Shape shape{array.shape(), array.shape() + array.ndim()};
    Dtype dtype = GetDtypeFromNumpyDtype(array.dtype());
    Strides strides{array.strides(), array.strides() + array.ndim()};

    int64_t first{};
    int64_t last{};
    std::tie(first, last) = GetDataRange(shape, strides, array.itemsize());
    py::buffer_info info = array.request();

    // Some backends may perform zero copy, so increment refcount of numpy ndarray not to be released in user codes.
    // Note that inc_ref() / dec_ref() is performed by the lambda capture.
    std::shared_ptr<void> data{static_cast<char*>(info.ptr) + first, [array](void*) {}};

    return MoveArrayBody(internal::FromHostData(shape, dtype, data, strides, -first, device));
}

namespace {

py::array MakeNumpyArrayFromArray(const py::module& m, const ArrayBodyPtr& self, bool copy) {
    Array array = Array{self}.ToNative();

    py::object dtype = GetNumpyDtypeFromModule(m, array.dtype());
    const Shape& shape = array.shape();
    const Strides& strides = array.strides();
    const void* ptr = internal::GetRawOffsetData(array);

    if (copy) {
        return py::array{dtype, shape, strides, ptr};
    }
    return py::array{dtype, shape, strides, ptr, py::cast(internal::MoveArrayBody(std::move(array)))};
}

// TODO(okapies): this is a workaround for improving performance
py::object MakeCupyArrayFromArray(const py::module& m, py::handle self) {
    Array array{py::cast<ArrayBodyPtr>(self)};
    Device& device = array.device();
    // TODO(okapies): rejects if array's device is not compatible with cupy

    py::object dtype = GetNumpyDtypeFromModule(m, array.dtype());
    const Shape& shape = array.shape();
    const Strides& strides = array.strides();

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const intptr_t ptr = reinterpret_cast<intptr_t>(array.raw_data());
    const auto range = GetDataRange(shape, strides, array.GetItemSize());
    const auto data_size = std::get<1>(range) - std::get<0>(range);
    const auto device_index = device.index();

    // Convert object to CuPy array using cupy.ndarray()
    auto memory_pointer = GetCachedCupyMemoryPointer();
    auto unowned_memory = GetCachedCupyUnownedMemory();
    py::object memptr = memory_pointer(unowned_memory(ptr, data_size, self, device_index), array.offset());

    auto ndarray = GetCachedCupyNdarray();
    return ndarray(ToTuple(shape), dtype, memptr, ToTuple(strides));
}

}  // namespace

ArrayBodyPtr MakeArray(py::handle object, py::handle dtype, bool copy, py::handle device) {
    nonstd::optional<Dtype> dtype_ = dtype.is_none() ? nonstd::nullopt : nonstd::optional<Dtype>(GetDtype(dtype));
    Device& dev = GetDevice(device);

    return MakeArray(object, dtype_, copy, dev);
}

ArrayBodyPtr MakeArray(py::handle object, const nonstd::optional<Dtype>& dtype, bool copy, Device& device) {
    // object is chainerx.ndarray
    if (py::isinstance<ArrayBody>(object)) {
        Array a = Array{py::cast<ArrayBodyPtr>(object)};
        Dtype dtype_ = dtype.has_value() ? *dtype : a.dtype();

        if (!copy && a.dtype() == dtype_ && &a.device() == &device) {
            return MoveArrayBody(std::move(a));
        }
        // Note that the graph is connected.
        if (&a.device() != &device) {
            return MoveArrayBody(a.ToDevice(device).AsType(dtype_, false));
        }
        if (a.dtype() != dtype_) {
            return MoveArrayBody(a.AsType(dtype_, true));
        }
        return MoveArrayBody(a.Copy());
    }

    // Convert object to NumPy array using numpy.array()
    // TODO(sonots): Remove dependency on numpy
    auto array_func = GetCachedNumpyArray();
    py::object dtype_name = py::none();
    if (dtype.has_value()) {
        dtype_name = py::str{GetDtypeName(*dtype)};
    }
    py::array np_array = array_func(object, py::arg("copy") = copy, py::arg("dtype") = dtype_name);

    // Convert NumPy array to ChainerX array
    return MakeArrayFromNumpyArray(np_array, device);
}

void InitChainerxArray(pybind11::module& m) {
    py::class_<ArrayBody, ArrayBodyPtr> c{m, "ndarray", py::buffer_protocol()};
    // TODO(hvy): Support all arguments in the constructor of numpy.ndarray.
    c.def(py::init([](const py::tuple& shape, py::handle dtype, py::handle device) {
              return MoveArrayBody(Empty(ToShape(shape), GetDtype(dtype), GetDevice(device)));
          }),
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device") = nullptr);
    c.def_property_readonly("__array_priority__", [](const ArrayBodyPtr & /*self*/) -> double { return 100.; });
    m.def("to_numpy",
          [m](const ArrayBodyPtr& array, bool copy) { return MakeNumpyArrayFromArray(m, array, copy); },
          py::arg("array"),
          py::arg("copy") = true);
    m.def("_to_cupy", [m](py::handle array) { return MakeCupyArrayFromArray(m, array); }, py::arg("array"));
    // This is currently for internal use (from Chainer) to support CuPy.
    // TODO(niboshi): Remove this once it will be possible to import cupy.ndarray using chx.array / chx.asarray.
    m.def("_fromrawpointer",
          [](intptr_t ptr,
             const py::tuple& shape,
             py::handle dtype,
             const py::tuple& strides,
             py::handle device,
             int64_t offset,
             py::object base) -> ArrayBodyPtr {
              // TODO(niboshi): Expose `base` as `ndarray.base` attribute.
              void* c_ptr = reinterpret_cast<void*>(ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
              // Note that inc_ref() / dec_ref() is performed by the lambda capture.
              std::shared_ptr<void> data{c_ptr, [base](void*) {}};
              return MoveArrayBody(FromData(ToShape(shape), GetDtype(dtype), data, ToStrides(strides), offset, GetDevice(device)));
          });
    c.def(py::pickle(
            [m](const ArrayBodyPtr& self) -> py::tuple { return py::make_tuple(MakeNumpyArrayFromArray(m, self, true), self->device()); },
            [](py::tuple state) -> ArrayBodyPtr {
                py::array numpy_array = state[0];
                Device& device = py::cast<Device&>(state[1]);
                return MakeArrayFromNumpyArray(numpy_array, device);
            }));
    c.def("__len__", [](const ArrayBodyPtr& self) -> size_t {
        // TODO(hvy): Do bounds cheking. For reference, Chainer throws an AttributeError.
        if (self->ndim() == 0) {
            throw pybind11::type_error("len() of unsized object");
        }
        return self->shape().front();
    });
    c.def("__bool__", [](const ArrayBodyPtr& self) -> bool { return static_cast<bool>(AsScalar(Array{self})); });
    c.def("__int__", [](const ArrayBodyPtr& self) -> int64_t { return static_cast<int64_t>(AsScalar(Array{self})); });
    c.def("__float__", [](const ArrayBodyPtr& self) -> double { return static_cast<double>(AsScalar(Array{self})); });
    // TODO(niboshi): Support arguments
    c.def("item", [](const ArrayBodyPtr& a) -> py::object {
        Scalar s = AsScalar(Array{a});
        switch (s.kind()) {
            case DtypeKind::kBool:
                return py::bool_{static_cast<bool>(s)};
            case DtypeKind::kInt:
                return py::int_{static_cast<int64_t>(s)};
            case DtypeKind::kFloat:
                return py::float_{static_cast<double>(s)};
            default:
                CHAINERX_NEVER_REACH();
        }
    });
    c.def("view", [](const ArrayBodyPtr& self) { return MoveArrayBody(Array{self}.MakeView()); });
    c.def("__repr__", [](const ArrayBodyPtr& self) { return Array{self}.ToString(); });
    c.def("to_device", [](const ArrayBodyPtr& self, py::handle device) { return MoveArrayBody(Array{self}.ToDevice(GetDevice(device))); });
    c.def("to_device", [](const ArrayBodyPtr& self, const std::string& backend_name, int index) {
        Device& device = GetDefaultContext().GetDevice({backend_name, index});
        return MoveArrayBody(Array{self}.ToDevice(device));
    });
    c.def("as_grad_stopped",
          [](const ArrayBodyPtr& self, bool copy) {
              return MoveArrayBody(Array{self}.AsGradStopped(copy ? CopyKind::kCopy : CopyKind::kView));
          },
          py::arg("copy") = false);
    c.def("as_grad_stopped",
          [](const ArrayBodyPtr& self, const std::vector<BackpropId>& backprop_ids, bool copy) {
              return MoveArrayBody(Array{self}.AsGradStopped(backprop_ids, copy ? CopyKind::kCopy : CopyKind::kView));
          },
          py::arg().noconvert(),
          py::arg("copy") = false);
    c.def("astype",
          [](const ArrayBodyPtr& self, py::handle dtype, bool copy) { return MoveArrayBody(Array{self}.AsType(GetDtype(dtype), copy)); },
          py::arg("dtype"),
          py::arg("copy") = true);
    c.def("copy", [](const ArrayBodyPtr& self) { return MoveArrayBody(Array{self}.Copy()); });
    c.def("__getitem__", [](const ArrayBodyPtr& self, py::handle key) { return MoveArrayBody(Array{self}.At(MakeArrayIndices(key))); });
    c.def("take",
          [](const ArrayBodyPtr& self, py::handle indices, const nonstd::optional<int8_t>& axis) {
              if (!axis.has_value()) {
                  throw NotImplementedError{"axis=None is not yet supported for chainerx.ndarray.take."};
              }
              if (py::isinstance<ArrayBody>(indices)) {
                  return MoveArrayBody(Array{self}.Take(Array{py::cast<ArrayBodyPtr>(indices)}, axis.value()));
              }
              if (py::isinstance<py::sequence>(indices)) {
                  nonstd::optional<Dtype> dtype = Dtype::kInt64;
                  return MoveArrayBody(Array{self}.Take(Array{MakeArray(indices, dtype, false, self->device())}, axis.value()));
              }
              if (py::isinstance<py::array>(indices)) {
                  return MoveArrayBody(
                          Array{self}.Take(Array{MakeArrayFromNumpyArray(py::cast<py::array>(indices), self->device())}, axis.value()));
              }
              throw py::type_error{"only integers, slices (`:`), sequence, numpy.ndarray and chainerx.newaxis (`None`) are valid indices"};
          },
          py::arg("indices"),
          py::arg("axis") = nullptr);
    c.def("transpose",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axes) {
              return MoveArrayBody(Array{self}.Transpose(ToAxes(axes)));
          },
          py::arg("axes") = nullptr);
    c.def("transpose", [](const ArrayBodyPtr& self, py::args args) { return MoveArrayBody(Array{self}.Transpose(ToAxes(args))); });
    c.def("reshape", [](const ArrayBodyPtr& self, py::tuple shape) { return MoveArrayBody(Array{self}.Reshape(ToShape(shape))); });
    c.def("reshape", [](const ArrayBodyPtr& self, const std::vector<int64_t>& shape) {
        return MoveArrayBody(Array{self}.Reshape({shape.begin(), shape.end()}));
    });
    c.def("reshape", [](const ArrayBodyPtr& self, py::args args) {
        if (args.size() == 0) {
            throw ChainerxError("Reshape takes exactly 1 argument (0 given).");
        }
        return MoveArrayBody(Array{self}.Reshape(ToShape(args)));
    });
    c.def("squeeze",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis) {
              return MoveArrayBody(Array{self}.Squeeze(ToAxes(axis)));
          },
          py::arg("axis") = nullptr);
    c.def("squeeze", [](const ArrayBodyPtr& self, int8_t axis) { return MoveArrayBody(Array{self}.Squeeze(Axes{axis})); }, py::arg("axis"));
    c.def("swapaxes",
          [](const ArrayBodyPtr& self, int8_t axis1, int8_t axis2) { return MoveArrayBody(Array{self}.Swapaxes(axis1, axis2)); },
          py::arg("axis1"),
          py::arg("axis2"));
    c.def("__eq__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} == Array{rhs}); },
          py::is_operator());
    c.def("__eq__",
          [](const ArrayBodyPtr& self, Scalar rhs) {
              // TODO(niboshi): More efficient implementation
              Array self_array{self};
              return MoveArrayBody(self_array == FullLike(self_array, rhs, self->device()));
          },
          py::is_operator());
    c.def("__ne__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} != Array{rhs}); },
          py::is_operator());
    c.def("__ne__",
          [](const ArrayBodyPtr& self, Scalar rhs) {
              // TODO(niboshi): More efficient implementation
              Array self_array{self};
              return MoveArrayBody(self_array != FullLike(self_array, rhs, self->device()));
          },
          py::is_operator());
    c.def("__gt__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} > Array{rhs}); },
          py::is_operator());
    c.def("__gt__",
          [](const ArrayBodyPtr& self, Scalar rhs) {
              // TODO(niboshi): More efficient implementation
              Array self_array{self};
              return MoveArrayBody(self_array > FullLike(self_array, rhs, self->device()));
          },
          py::is_operator());
    c.def("__ge__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} >= Array{rhs}); },
          py::is_operator());
    c.def("__ge__",
          [](const ArrayBodyPtr& self, Scalar rhs) {
              // TODO(niboshi): More efficient implementation
              Array self_array{self};
              return MoveArrayBody(self_array >= FullLike(self_array, rhs, self->device()));
          },
          py::is_operator());
    c.def("__lt__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} < Array{rhs}); },
          py::is_operator());
    c.def("__lt__",
          [](const ArrayBodyPtr& self, Scalar rhs) {
              // TODO(niboshi): More efficient implementation
              Array self_array{self};
              return MoveArrayBody(self_array < FullLike(self_array, rhs, self->device()));
          },
          py::is_operator());
    c.def("__le__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} <= Array{rhs}); },
          py::is_operator());
    c.def("__le__",
          [](const ArrayBodyPtr& self, Scalar rhs) {
              // TODO(niboshi): More efficient implementation
              Array self_array{self};
              return MoveArrayBody(self_array <= FullLike(self_array, rhs, self->device()));
          },
          py::is_operator());
    c.def("__neg__", [](const ArrayBodyPtr& self) { return MoveArrayBody(-Array{self}); });
    c.def("__abs__", [](const ArrayBodyPtr& self) { return MoveArrayBody(Absolute(Array{self})); }, py::is_operator());
    c.def("__iadd__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(std::move(Array{self} += Array{rhs})); },
          py::is_operator());
    c.def("__iadd__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(std::move(Array{self} += rhs)); }, py::is_operator());
    c.def("__isub__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(std::move(Array{self} -= Array{rhs})); },
          py::is_operator());
    c.def("__isub__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(std::move(Array{self} -= rhs)); }, py::is_operator());
    c.def("__imul__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(std::move(Array{self} *= Array{rhs})); },
          py::is_operator());
    c.def("__imul__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(std::move(Array{self} *= rhs)); }, py::is_operator());
    c.def("__ifloordiv__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) {
              internal::IFloorDivide(Array{self}, Array{rhs});
              return self;
          },
          py::is_operator());
    c.def("__ifloordiv__", [](const ArrayBodyPtr& self, Scalar rhs) {
        internal::IFloorDivide(Array{self}, rhs);
        return self;
    });
    c.def("__itruediv__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(std::move(Array{self} /= Array{rhs})); },
          py::is_operator());
    c.def("__itruediv__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(std::move(Array{self} /= rhs)); });
    c.def("__iand__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(std::move(Array{self} &= Array{rhs})); },
          py::is_operator());
    c.def("__iand__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(std::move(Array{self} &= rhs)); }, py::is_operator());
    c.def("__ior__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(std::move(Array{self} |= Array{rhs})); },
          py::is_operator());
    c.def("__ior__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(std::move(Array{self} |= rhs)); }, py::is_operator());
    c.def("__ixor__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(std::move(Array{self} ^= Array{rhs})); },
          py::is_operator());
    c.def("__ixor__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(std::move(Array{self} ^= rhs)); }, py::is_operator());
    c.def("__add__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} + Array{rhs}); },
          py::is_operator());
    c.def("__add__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(Array{self} + rhs); }, py::is_operator());
    c.def("__radd__", [](const ArrayBodyPtr& self, Scalar lhs) { return MoveArrayBody(lhs + Array{self}); }, py::is_operator());
    c.def("__sub__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} - Array{rhs}); },
          py::is_operator());
    c.def("__sub__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(Array{self} - rhs); }, py::is_operator());
    c.def("__rsub__", [](const ArrayBodyPtr& self, Scalar lhs) { return MoveArrayBody(lhs - Array{self}); }, py::is_operator());
    c.def("__mul__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} * Array{rhs}); },
          py::is_operator());
    c.def("__mul__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(Array{self} * rhs); }, py::is_operator());
    c.def("__rmul__", [](const ArrayBodyPtr& self, Scalar lhs) { return MoveArrayBody(lhs * Array{self}); }, py::is_operator());
    c.def("__floordiv__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(FloorDivide(Array{self}, Array{rhs})); },
          py::is_operator());
    c.def("__floordiv__",
          [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(FloorDivide(Array{self}, rhs)); },
          py::is_operator());
    c.def("__rfloordiv__",
          [](const ArrayBodyPtr& self, Scalar lhs) { return MoveArrayBody(FloorDivide(lhs, Array{self})); },
          py::is_operator());
    c.def("__truediv__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} / Array{rhs}); },
          py::is_operator());
    c.def("__truediv__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(Array{self} / rhs); }, py::is_operator());
    c.def("__pow__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Power(Array{self}, Array{rhs})); },
          py::is_operator());
    c.def("__pow__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(Power(Array{self}, rhs)); }, py::is_operator());
    c.def("__rpow__", [](const ArrayBodyPtr& self, Scalar lhs) { return MoveArrayBody(Power(lhs, Array{self})); }, py::is_operator());

    c.def("__rtruediv__", [](const ArrayBodyPtr& self, Scalar lhs) { return MoveArrayBody(lhs / Array{self}); }, py::is_operator());
    c.def("__and__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} & Array{rhs}); },
          py::is_operator());
    c.def("__and__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(Array{self} & rhs); }, py::is_operator());
    c.def("__rand__", [](const ArrayBodyPtr& self, Scalar lhs) { return MoveArrayBody(Array{self} & lhs); }, py::is_operator());
    c.def("__or__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} | Array{rhs}); },
          py::is_operator());
    c.def("__or__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(Array{self} | rhs); }, py::is_operator());
    c.def("__ror__", [](const ArrayBodyPtr& self, Scalar lhs) { return MoveArrayBody(Array{self} | lhs); }, py::is_operator());
    c.def("__xor__",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return MoveArrayBody(Array{self} ^ Array{rhs}); },
          py::is_operator());
    c.def("__xor__", [](const ArrayBodyPtr& self, Scalar rhs) { return MoveArrayBody(Array{self} ^ rhs); }, py::is_operator());
    c.def("__rxor__", [](const ArrayBodyPtr& self, Scalar lhs) { return MoveArrayBody(Array{self} ^ lhs); }, py::is_operator());
    c.def("sum",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) { return MoveArrayBody(Array{self}.Sum(Axes{axis}, keepdims)); },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("sum",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Array{self}.Sum(ToAxes(axis), keepdims));
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("max",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) { return MoveArrayBody(Array{self}.Max(Axes{axis}, keepdims)); },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("max",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Array{self}.Max(ToAxes(axis), keepdims));
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("min",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) { return MoveArrayBody(Array{self}.Min(Axes{axis}, keepdims)); },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("min",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Array{self}.Min(ToAxes(axis), keepdims));
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("mean",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) { return MoveArrayBody(Array{self}.Mean(Axes{axis}, keepdims)); },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("mean",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Array{self}.Mean(ToAxes(axis), keepdims));
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("var",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) { return MoveArrayBody(Array{self}.Var(Axes{axis}, keepdims)); },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("var",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Array{self}.Var(ToAxes(axis), keepdims));
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("all",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) { return MoveArrayBody(Array{self}.All(Axes{axis}, keepdims)); },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("all",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Array{self}.All(ToAxes(axis), keepdims));
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("any",
          [](const ArrayBodyPtr& self, int8_t axis, bool keepdims) { return MoveArrayBody(Array{self}.Any(Axes{axis}, keepdims)); },
          py::arg("axis"),
          py::arg("keepdims") = false);
    c.def("any",
          [](const ArrayBodyPtr& self, const nonstd::optional<std::vector<int8_t>>& axis, bool keepdims) {
              return MoveArrayBody(Array{self}.Any(ToAxes(axis), keepdims));
          },
          py::arg("axis") = nullptr,
          py::arg("keepdims") = false);
    c.def("argmax",
          [](const ArrayBodyPtr& self, const nonstd::optional<int8_t>& axis) { return MoveArrayBody(ArgMax(Array{self}, ToAxes(axis))); },
          py::arg("axis") = nullptr);
    c.def("argmin",
          [](const ArrayBodyPtr& self, const nonstd::optional<int8_t>& axis) { return MoveArrayBody(ArgMin(Array{self}, ToAxes(axis))); },
          py::arg("axis") = nullptr);
    c.def("dot", [](const ArrayBodyPtr& self, const ArrayBodyPtr& b) { return MoveArrayBody(Array{self}.Dot(Array{b})); }, py::arg("b"));
    c.def("fill",
          [](const ArrayBodyPtr& self, Scalar value) {
              Array{self}.Fill(value);
              return;
          },
          py::arg("value"));

    c.def("require_grad",
          [](const ArrayBodyPtr& self, const nonstd::optional<BackpropId>& backprop_id) {
              return MoveArrayBody(std::move(Array{self}.RequireGrad(backprop_id)));
          },
          py::arg("backprop_id") = nullptr);
    c.def("is_grad_required",
          [](const ArrayBodyPtr& self, const nonstd::optional<BackpropId>& backprop_id) { return Array{self}.IsGradRequired(backprop_id); },
          py::arg("backprop_id") = nullptr);
    c.def("is_backprop_required",
          [](const ArrayBodyPtr& self, const nonstd::optional<BackpropId>& backprop_id) {
              return Array{self}.IsBackpropRequired(backprop_id);
          },
          py::arg("backprop_id") = nullptr);
    c.def("is_backprop_required",
          [](const ArrayBodyPtr& self, AnyGraph any_graph) { return Array{self}.IsBackpropRequired(any_graph); },
          py::arg("backprop_id"));
    c.def("get_grad",
          [](const ArrayBodyPtr& self, const nonstd::optional<BackpropId>& backprop_id) -> ConstArrayBodyPtr {
              const nonstd::optional<Array>& grad = Array{self}.GetGrad(backprop_id);
              if (!grad.has_value()) {
                  return nullptr;
              }
              return internal::GetArrayBody(*grad);
          },
          py::arg("backprop_id") = nullptr);
    c.def("set_grad",
          [](const ArrayBodyPtr& self, const ArrayBodyPtr& grad, const nonstd::optional<BackpropId>& backprop_id) {
              Array array{self};
              if (grad) {
                  array.SetGrad(Array{grad}, backprop_id);
              } else {
                  array.ClearGrad(backprop_id);
              }
          },
          py::arg("grad"),
          py::arg("backprop_id") = nullptr);
    c.def("backward",
          [](const ArrayBodyPtr& self, const nonstd::optional<BackpropId>& backprop_id, bool enable_double_backprop) {
              auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
              Backward(Array{self}, backprop_id, double_backprop);
          },
          py::arg("backprop_id") = nullptr,
          py::arg("enable_double_backprop") = false);
    c.def("_debug_dump_computational_graph",
          [](const ArrayBodyPtr& self, const nonstd::optional<BackpropId>& backprop_id) {
              DebugDumpComputationalGraph(std::cout, Array{self}, backprop_id);
          },
          py::arg("backprop_id") = nullptr);
    c.def_property(
            "grad",
            [](const ArrayBodyPtr& self) -> ConstArrayBodyPtr {
                const nonstd::optional<Array>& grad = Array{self}.GetGrad(nonstd::nullopt);
                if (!grad.has_value()) {
                    return nullptr;
                }
                return internal::GetArrayBody(*grad);
            },
            [](const ArrayBodyPtr& self, const ArrayBodyPtr& grad) {
                Array array{self};
                if (grad) {
                    array.SetGrad(Array{grad}, nonstd::nullopt);
                } else {
                    array.ClearGrad(nonstd::nullopt);
                }
            });
    c.def("cleargrad",
          [](const ArrayBodyPtr& self, const nonstd::optional<BackpropId>& backprop_id) { Array{self}.ClearGrad(backprop_id); },
          py::arg("backprop_id") = nullptr);
    c.def_property_readonly(
            "device", [](const ArrayBodyPtr& self) -> Device& { return self->device(); }, py::return_value_policy::reference);
    c.def_property_readonly("dtype", [m](const ArrayBodyPtr& self) { return GetNumpyDtypeFromModule(m, self->dtype()); });
    c.def_property_readonly("itemsize", [](const ArrayBodyPtr& self) { return self->GetItemSize(); });
    c.def_property_readonly("is_contiguous", [](const ArrayBodyPtr& self) { return self->IsContiguous(); });
    c.def_property_readonly("ndim", [](const ArrayBodyPtr& self) { return self->ndim(); });
    c.def_property_readonly("offset", [](const ArrayBodyPtr& self) { return self->offset(); });
    c.def_property_readonly("shape", [](const ArrayBodyPtr& self) { return ToTuple(self->shape()); });
    c.def_property_readonly("strides", [](const ArrayBodyPtr& self) { return ToTuple(self->strides()); });
    c.def_property_readonly("nbytes", [](const ArrayBodyPtr& self) { return self->GetNBytes(); });
    c.def_property_readonly("size", [](const ArrayBodyPtr& self) { return self->GetTotalSize(); });
    c.def_property_readonly("T", [](const ArrayBodyPtr& self) { return MoveArrayBody(Array{self}.Transpose()); });
    // Returns the data address, before adding offset.
    // TODO(niboshi): Consider what to do with the backends in which the "pointer" is not available from host.
    c.def_property_readonly("data_ptr", [](const ArrayBodyPtr& self) -> intptr_t {
        return reinterpret_cast<intptr_t>(self->data().get());  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    });
    c.def_property_readonly("data_size", [](const ArrayBodyPtr& self) -> int64_t {
        auto range = GetDataRange(self->shape(), self->strides(), self->GetItemSize());
        return std::get<1>(range) - std::get<0>(range);
    });
    // TODO(niboshi): Remove this in favor of data_ptr.
    c.def_property_readonly(
            "_debug_data_memory_address",  // These methods starting with `_debug_` are stubs for testing
            [](const ArrayBodyPtr& self) -> intptr_t {
                const void* ptr = self->data().get();
                return reinterpret_cast<intptr_t>(ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
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
                T value = native::StorageToDataType<const T>(iarray[it]);
                if (std::is_same<T, chainerx::Float16>::value) {
                    list.append(static_cast<double>(value));
                } else {
                    list.append(value);
                }
            }
        });

        return list;
    });
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
