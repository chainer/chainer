#include "xchainer/python/array.h"

#include <algorithm>

#include "xchainer/array.h"

namespace xchainer {

namespace py = pybind11;

void InitXchainerArray(pybind11::module& m) {
    py::class_<Array, std::shared_ptr<Array>>{m, "Array"}
        .def(py::init([](const Shape& shape, Dtype dtype, py::list list) {
            auto total_size = shape.total_size();
            auto bytes = GetElementSize(dtype) * total_size;
            if (static_cast<size_t>(total_size) != list.size()) {
                throw DimensionError("Invalid data length");
            }

            std::shared_ptr<void> ptr = std::make_unique<uint8_t[]>(bytes);
            auto func = [&](auto dummy) {
                using T = decltype(dummy);
                std::transform(list.begin(), list.end(), reinterpret_cast<T*>(ptr.get()), [](auto& item) { return py::cast<T>(item); });
            };
            switch (dtype) {
                case Dtype::kBool:
                    func(static_cast<bool>(0));
                    break;
                case Dtype::kInt8:
                    func(static_cast<int8_t>(0));
                    break;
                case Dtype::kInt16:
                    func(static_cast<int16_t>(0));
                    break;
                case Dtype::kInt32:
                    func(static_cast<int32_t>(0));
                    break;
                case Dtype::kInt64:
                    func(static_cast<int64_t>(0));
                    break;
                case Dtype::kUInt8:
                    func(static_cast<uint8_t>(0));
                    break;
                case Dtype::kFloat32:
                    func(static_cast<float>(0));
                    break;
                case Dtype::kFloat64:
                    func(static_cast<double>(0));
                    break;
                default:
                    assert(0);
            }
            return Array{shape, dtype, ptr};
        }))
        .def_property_readonly("dtype", &Array::dtype)
        .def_property_readonly("shape", &Array::shape)
        .def_property_readonly("is_contiguous", &Array::is_contiguous)
        .def_property_readonly("total_size", &Array::total_size)
        .def_property_readonly("element_bytes", &Array::element_bytes)
        .def_property_readonly("total_bytes", &Array::total_bytes)
        .def_property_readonly("offset", &Array::offset)
        .def_property_readonly("debug_flat_data", [](const Array& self) {  // This method is a stub for testing
            py::list list;
            auto size = self.total_size();
            auto func = [&](auto dummy) {
                using T = decltype(dummy);
                const T& data = *std::static_pointer_cast<const T>(self.data());
                for (int64_t i = 0; i < size; ++i) {
                    list.append((&data)[i]);
                }
            };
            switch (self.dtype()) {
                case Dtype::kBool:
                    func(static_cast<bool>(0));
                    break;
                case Dtype::kInt8:
                    func(static_cast<int8_t>(0));
                    break;
                case Dtype::kInt16:
                    func(static_cast<int16_t>(0));
                    break;
                case Dtype::kInt32:
                    func(static_cast<int32_t>(0));
                    break;
                case Dtype::kInt64:
                    func(static_cast<int64_t>(0));
                    break;
                case Dtype::kUInt8:
                    func(static_cast<uint8_t>(0));
                    break;
                case Dtype::kFloat32:
                    func(static_cast<float>(0));
                    break;
                case Dtype::kFloat64:
                    func(static_cast<double>(0));
                    break;
                default:
                    assert(0);
            }
            return list;
        })
        .def("__add__", [](const Array& lhs, const Array& rhs) { return lhs.Add(rhs); })
        .def("__iadd__", [](Array& lhs, const Array& rhs) { return lhs.IAdd(rhs); })
        .def("__mul__", [](const Array& lhs, const Array& rhs) { return lhs.Mul(rhs); })
        .def("__imul__", [](Array& lhs, const Array& rhs) { return lhs.IMul(rhs); });
}

}  // namespace xchainer
