#include "xchainer/python/shape.h"

#include <algorithm>
#include <vector>

namespace xchainer {

namespace py = pybind11;

Shape ToShape(const py::tuple& tup) {
    std::vector<int64_t> v;
    v.reserve(tup.size());
    std::transform(tup.begin(), tup.end(), std::back_inserter(v), [](auto& item) { return py::cast<int64_t>(item); });
    return Shape{v};
}

py::tuple ToTuple(const Shape& shape) {
    // TODO(beam2d): Consider casting directly to tuple instead of casting to list.
    py::list ret;
    for (int64_t dim : shape) {
        ret.append(dim);
    }
    return ret;
}

}  // namespace xchainer
