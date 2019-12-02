#include "chainerx/python/common_export.h"

#include "chainerx/python/shape.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "chainerx/shape.h"

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
            throw py::type_error{"shape not understood: " + py::cast<std::string>(py::repr(shape))};
        }
        return Shape{seq};
    }
    if (py::isinstance<py::int_>(shape)) {
        return Shape{shape.cast<int64_t>()};
    }
    throw py::type_error{"expected sequence object or a single integer"};
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
