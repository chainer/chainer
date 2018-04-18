#include "xchainer/python/shape.h"

#include <algorithm>

#include "xchainer/ndim_vector.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;

Shape ToShape(const py::tuple& tup) {
    NdimVector<int64_t> v{};
    std::transform(tup.begin(), tup.end(), std::back_inserter(v), [](auto& item) { return py::cast<int64_t>(item); });
    return Shape{v.begin(), v.end()};
}

py::tuple ToTuple(const Shape& shape) {
    py::tuple ret{shape.size()};
    for (size_t i = 0; i < shape.size(); ++i) {
        ret[i] = shape[i];
    }
    return ret;
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
