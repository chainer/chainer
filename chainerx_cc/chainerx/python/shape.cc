#include "chainerx/python/shape.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

Shape ToShape(py::handle shape) {
    if (py::isinstance<py::sequence>(shape)) {
        std::vector<int64_t> seq{};
        try {
            seq = py::cast<std::vector<int64_t>>(shape);
        } catch (const py::cast_error& e) {
            throw py::type_error{std::string{"shape not understood: "} + py::cast<std::string>(py::repr(shape))};
        }
        return Shape{seq};
    } else if (py::isinstance<py::int_>(shape)) {
        return Shape{shape.cast<int64_t>()};
    } else {
        throw py::type_error{"only integer and sequence are valid shape"};
    }
}

py::tuple ToTuple(const Shape& shape) {
    py::tuple ret{shape.size()};
    for (size_t i = 0; i < shape.size(); ++i) {
        ret[i] = shape[i];
    }
    return ret;
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
