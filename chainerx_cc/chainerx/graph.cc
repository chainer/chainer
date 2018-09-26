#include "chainerx/graph.h"

#include <ostream>
#include <string>

#include "chainerx/context.h"

namespace chainerx {

std::ostream& operator<<(std::ostream& os, const BackpropId& backprop_id) { return os << backprop_id.GetName(); }

std::string BackpropId::GetName() const { return context_.get().GetBackpropName(*this); }

void BackpropId::CheckValid() const { context_.get().CheckValidBackpropId(*this); }

}  // namespace chainerx
