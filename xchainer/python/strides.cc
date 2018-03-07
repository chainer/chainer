#include "xchainer/python/strides.h"

#include <pybind11/operators.h>

#include "xchainer/strides.h"

#include "xchainer/python/common.h"

namespace xchainer {

namespace py = pybind11;

void InitXchainerStrides(pybind11::module& m) {
    py::class_<Strides>{m, "Strides"}
        .def(py::init([](py::tuple tup) {  // __init__ by a tuple
            std::vector<int64_t> v;
            std::transform(tup.begin(), tup.end(), std::back_inserter(v), [](auto& item) { return py::cast<int64_t>(item); });
            return Strides(v);
        }))
        .def(py::self == py::self)  // NOLINT
        .def("__eq__",              // Equality with a tuple
             [](const Strides& self, const py::tuple& tup) {
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
        .def("__repr__", static_cast<std::string (Strides::*)() const>(&Strides::ToString))
        .def_property_readonly("ndim", &Strides::ndim);

    py::implicitly_convertible<py::tuple, Strides>();
}

}  // namespace xchainer
