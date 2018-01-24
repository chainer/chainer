#pragma once

#include "xchainer/backend.h"

namespace xchainer {
namespace cuda {

class CudaBackend : public Backend {
public:
    void Synchronize();
};

}  // namespace cuda
}  // namespace xchainer
