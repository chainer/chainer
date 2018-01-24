#pragma onnce

#include "xchainer/backend.h"

namespace xchainer {

class NativeBackend : public Backend {
public:
    void Synchronize();
};

}  // namespace xchainer
