#include "xchainer/python/strides.h"

#include <algorithm>

#include "xchainer/ndim_vector.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

Strides ToStrides(const py::tuple& tup) {
    NdimVector<int64_t> v{};
    std::transform(tup.begin(), tup.end(), std::back_inserter(v), [](auto& item) { return py::cast<int64_t>(item); });
    return Strides{v.begin(), v.end()};
}

py::tuple ToTuple(const Strides& strides) {
    py::tuple ret{strides.size()};
    for (size_t i = 0; i < strides.size(); ++i) {
        ret[i] = strides[i];
    }
    return ret;
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
