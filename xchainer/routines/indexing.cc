#include "xchainer/routines/indexing.h"

#include <vector>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/dtype.h"
#include "xchainer/graph_id.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace routines {
namespace {

// Returns an array where elements at indices are added by the addends `b`.
//
// The original values of this array are not altered.
Array AddAt(const Array& a, const std::vector<ArrayIndex>& indices, const Array& b) {
    // TODO(sonots): dtype conversion
    CheckEqual(a.dtype(), b.dtype());

    Array out = a.AsConstant(CopyKind::kCopy);
    Array out_view = out.At(indices);

    // TODO(sonots): broadcasting
    CheckEqual(out_view.shape(), b.shape());

    a.device().Add(b, out_view, out_view);

    auto this_backward_function = [](const Array& gout, const std::vector<GraphId>&) { return gout; };
    auto addend_backward_function = [indices](const Array& gout, const std::vector<GraphId>&) { return gout.At(indices); };
    internal::SetUpOpNodes("add_at", {a, b}, out, {this_backward_function, addend_backward_function});

    return out;
}

}  // namespace
}  // namespace routines
}  // namespace xchainer
