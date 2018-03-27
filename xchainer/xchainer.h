#pragma once

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/indexing.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/routines/math.h"

namespace xchainer {
// TODO(sonots): Create python binding and do tests

// creation
using routines::Copy;
using routines::Empty;
using routines::EmptyLike;
using routines::FromBuffer;
using routines::Full;
using routines::FullLike;
using routines::Ones;
using routines::OnesLike;
using routines::Zeros;
using routines::ZerosLike;

// manipulation
using routines::BroadcastTo;
using routines::Reshape;
using routines::Squeeze;
using routines::Transpose;

// math
using routines::Add;
using routines::Mul;
using routines::Sum;

}  // namespace xchainer
