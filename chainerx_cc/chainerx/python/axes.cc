#include "chainerx/python/common_export.h"

#include "chainerx/python/axes.h"

#include <algorithm>
#include <cstdint>

#include <pybind11/pybind11.h>

#include "chainerx/axes.h"

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

Axes ToAxes(const py::tuple& tup) {
    Axes axes{};
    std::transform(tup.begin(), tup.end(), std::back_inserter(axes), [](auto& item) { return py::cast<int64_t>(item); });
    return axes;
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
