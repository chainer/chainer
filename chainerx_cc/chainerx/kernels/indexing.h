#pragma once

#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/op.h"

namespace chainerx {

class AddAtOp : public Op {
public:
    static const char* name() { return "AddAt"; }

    virtual void Call(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) = 0;
};

class TakeOp : public Op {
public:
    static const char* name() { return "Take"; }

    virtual void Call(const Array& a, const Array& indices, int8_t axis, const Array& out) = 0;
};

}  // namespace chainerx
