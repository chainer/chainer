#include "xchainer/python/array_index.h"

#include "xchainer/array_index.h"
#include "xchainer/slice.h"

#include "xchainer/python/common.h"

#include <iostream>

namespace xchainer {

namespace py = pybind11;

void InitXchainerArrayIndex(pybind11::module& m) {
    py::class_<ArrayIndex>{m, "ArrayIndex"}
            .def(py::init([](py::int_ index) { return ArrayIndex(py::cast<int64_t>(index)); }))
            .def(py::init([](py::slice slice) {
                std::cout << py::cast<int64_t>(((PySliceObject*)(slice.ptr()))->start) << std::endl;
                // Slice slice_{py::cast<int64_t>(slice.start()), py::cast<int64_t>(slice.stop()), py::cast<int64_t>(slice.step())};
                // return ArrayIndex(slice_);
                return 1;
            }));

    //.def(py::init([](py::tuple tup) {  // __init__ by a tuple
    //    std::vector<int64_t> v;
    //    std::transform(tup.begin(), tup.end(), std::back_inserter(v), [](auto& item) { return py::cast<int64_t>(item); });
    //    return Shape(v);
    //}))
    //.def(py::self == py::self)  // NOLINT
    //.def("__eq__",              // Equality with a tuple
    //     [](const Shape& self, const py::tuple& tup) {
    //         if (static_cast<size_t>(self.ndim()) != tup.size()) {
    //             return false;
    //         }
    //         try {
    //             return std::equal(self.begin(), self.end(), tup.begin(), tup.end(), [](const auto& dim, const auto& item) {
    //                 int64_t dim2 = py::cast<int64_t>(item);
    //                 return dim == dim2;
    //             });
    //         } catch (const py::cast_error& e) {
    //             return false;
    //         }
    //     })
    //.def("__repr__", static_cast<std::string (Shape::*)() const>(&Shape::ToString))
    //.def_property_readonly("ndim", &Shape::ndim)
    //.def_property_readonly("total_size", &Shape::GetTotalSize);

    // py::implicitly_convertible<py::tuple, Shape>();
}

}  // namespace xchainer
