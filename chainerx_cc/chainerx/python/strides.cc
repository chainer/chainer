#include "chainerx/python/common_export.h"

#include "chainerx/python/strides.h"

#include <algorithm>
#include <cstdint>

namespace chainerx {
namespace python {
namespace python_internal {

namespace py = pybind11;

Strides ToStrides(const py::tuple& tup) {
    Strides strides{};
    std::transform(tup.begin(), tup.end(), std::back_inserter(strides), [](auto& item) { return py::cast<int64_t>(item); });
    return strides;
}

py::tuple ToTuple(const Strides& strides) {
    py::tuple ret{strides.size()};
    for (size_t i = 0; i < strides.size(); ++i) {
        ret[i] = strides[i];
    }
    return ret;
}

}  // namespace python_internal
}  // namespace python
}  // namespace chainerx
