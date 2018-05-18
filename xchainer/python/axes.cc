#include "xchainer/python/axes.h"

#include <algorithm>
#include <cstdint>

#include <pybind11/pybind11.h>

#include "xchainer/axes.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

Axes ToAxes(const py::tuple& tup) {
    Axes axes{};
    std::transform(tup.begin(), tup.end(), std::back_inserter(axes), [](auto& item) { return py::cast<int64_t>(item); });
    return axes;
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
