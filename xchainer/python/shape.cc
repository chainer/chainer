#include "xchainer/python/shape.h"

#include <algorithm>

#include <pybind11/operators.h>

#include "xchainer/shape.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;

void InitXchainerShape(pybind11::module& m) {
    py::class_<Shape>{m, "Shape"}
        .def(py::init([](py::tuple tup) {  // __init__ by a tuple
            std::vector<int64_t> v;
            std::transform(tup.begin(), tup.end(), std::back_inserter(v), [](auto& item) { return py::cast<int64_t>(item); });
            return Shape(v);
        }))
        .def(py::self == py::self)
        .def("__eq__",  // Equality with a tuple
             [](const Shape& self, const py::tuple& tup) {
                 if (static_cast<size_t>(self.ndim()) != tup.size()) {
                     return false;
                 }
                 try {
                     return std::equal(self.begin(), self.end(), tup.begin(), tup.end(), [](const auto& dim, const auto& item) {
                         int64_t dim2 = py::cast<int64_t>(item);
                         return dim == dim2;
                     });
                 } catch (const py::cast_error& e) {
                     return false;
                 }
             })
        .def("__repr__", static_cast<std::string (Shape::*)() const>(&Shape::ToString))
        .def_property_readonly("ndim", &Shape::ndim)
        .def_property_readonly("size", &Shape::size)
        .def_property_readonly("total_size", &Shape::GetTotalSize);

    py::implicitly_convertible<py::tuple, Shape>();
}

}  // namespace xchainer
