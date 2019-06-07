#include "chainerx/python/shape.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "chainerx/shape.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

Shape ToShape(const py::handle& object) {
    if (py::isinstance<py::sequence>(object)) {
        Shape shape{};
        py::sequence& seq = object;
        std::transform(seq.begin(), seq.end(), std::back_inserter(shape), [](auto& item) { return py::cast<int64_t>(item); });
        return shape;
    } else if (py::isinstance<py::int_>(object)) {
        return Shape{object.cast<int64_t>()};
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
