#include "chainerx/routines/pooling.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <absl/types/optional.h>

#include "chainerx/array.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward_builder.h"
#include "chainerx/backward_context.h"
#include "chainerx/constant.h"
#include "chainerx/device.h"
#include "chainerx/dims.h"
#include "chainerx/error.h"
#include "chainerx/graph.h"
#include "chainerx/kernels/pooling.h"
#include "chainerx/routines/arithmetic.h"
#include "chainerx/routines/routines_util.h"

namespace chainerx {
namespace {

void CheckPoolInputs(const Array& x, const Dims& kernel_size, const Dims& stride, const Dims& pad) {
    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions.
    if (static_cast<int8_t>(kernel_size.size()) != ndim) {
        throw DimensionError{"Wrong numbers of kernel size dimensions ", kernel_size.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(stride.size()) != ndim) {
        throw DimensionError{"Wrong numbers of strides ", stride.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(pad.size()) != ndim) {
        throw DimensionError{"Wrong numbers of paddings ", pad.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (ndim == 0) {
        throw DimensionError{"Pooling operation requires at least one spatial dimension."};
    }
    if (std::any_of(kernel_size.begin(), kernel_size.end(), [](int64_t ks) { return ks <= 0; })) {
        throw DimensionError{"Kernel size elements must be greater than 0: ", DimsFormatter{kernel_size}, "."};
    }
    if (std::any_of(stride.begin(), stride.end(), [](int64_t s) { return s <= 0; })) {
        throw DimensionError{"Stride elements must be greater than 0: ", DimsFormatter{stride}, "."};
    }
}

}  // namespace

Array MaxPool(const Array& x, const Dims& kernel_size, const Dims& stride, const Dims& pad, bool cover_all) {
    CheckPoolInputs(x, kernel_size, stride, pad);

    Array out{};
    std::shared_ptr<MaxPoolGradState> state{};
    {
        NoBackpropModeScope scope{};
        std::tie(out, state) =
                x.device().backend().CallKernel<MaxPoolKernel>(x.AsGradStopped(), kernel_size, stride, pad, cover_all, true, absl::nullopt);
    }
    internal::MakeViewForForwardBackwardOutput(out);

    // Supporting arbitrary number of backwards using a recursive definition.
    // TODO(hvy): Test backward of double backward.
    struct MaxPoolBwd {
        void operator()(BackwardContext& bctx1) {
            const Array& gout = *bctx1.output_grad();

            Array gx{};
            std::shared_ptr<MaxPoolGradGradState> grad_grad_state{};
            {
                NoBackpropModeScope scope{};
                std::tie(gx, grad_grad_state) = gout.device().backend().CallKernel<MaxPoolGradKernel>(
                        gout.AsGradStopped(), kernel_size, stride, pad, state, true, absl::nullopt);
            }
            internal::MakeViewForForwardBackwardOutput(gx);

            {
                BackwardBuilder bb2{"max_pooling_backward", gout, gx};
                if (BackwardBuilder::Target bt2 = bb2.CreateTarget(0)) {
                    bt2.Define([st = *this, grad_grad_state = std::move(grad_grad_state)](BackwardContext& bctx2) {
                        const Array& ggx = *bctx2.output_grad();

                        Array ggout{};
                        {
                            NoBackpropModeScope scope{};
                            ggout = ggx.device().backend().CallKernel<MaxPoolGradGradKernel>(
                                    ggx.AsGradStopped(), st.kernel_size, st.stride, st.pad, st.cover_all, grad_grad_state, absl::nullopt);
                        }

                        // Make ggout further backpropable.
                        {
                            BackwardBuilder bb3{"max_pooling_double_backward", ggx, ggout};
                            if (BackwardBuilder::Target bt3 = bb3.CreateTarget(0)) {
                                bt3.Define(st);
                            }
                            bb3.Finalize();
                        }
                        bctx2.input_grad() = std::move(ggout);
                    });
                }
                bb2.Finalize();
            }
            bctx1.input_grad() = std::move(gx);
        }

        Dims kernel_size;
        Dims stride;
        Dims pad;
        bool cover_all;
        std::shared_ptr<MaxPoolGradState> state;
    };

    {
        BackwardBuilder bb1{"max_pooling", x, out};
        if (BackwardBuilder::Target bt1 = bb1.CreateTarget(0)) {
            bt1.Define(MaxPoolBwd{kernel_size, stride, pad, cover_all, std::move(state)});
        }
        bb1.Finalize();
    }
    return out;
}

Array AveragePool(const Array& x, const Dims& kernel_size, const Dims& stride, const Dims& pad, AveragePoolPadMode pad_mode) {
    if (GetKind(x.dtype()) != DtypeKind::kFloat) {
        throw DtypeError("cannot apply average pooling to ", x.dtype(), " array (floatXX array is expected)");
    }
    CheckPoolInputs(x, kernel_size, stride, pad);

    Array out{};
    std::shared_ptr<AveragePoolGradState> state{};
    {
        NoBackpropModeScope scope{};
        std::tie(out, state) = x.device().backend().CallKernel<AveragePoolKernel>(
                x.AsGradStopped(), kernel_size, stride, pad, pad_mode, true, absl::nullopt);
    }
    internal::MakeViewForForwardBackwardOutput(out);

    {
        BackwardBuilder bb1{"average_pool", x, out};
        if (BackwardBuilder::Target bt1 = bb1.CreateTarget(0)) {
            bt1.Define([state = std::move(state), kernel_size, stride, pad, pad_mode](BackwardContext& bctx) {
                const Array& gout = *bctx.output_grad();

                Array gx{};
                {
                    NoBackpropModeScope scope{};
                    gx = gout.device().backend().CallKernel<AveragePoolGradKernel>(
                            gout.AsGradStopped(), kernel_size, stride, pad, pad_mode, state, absl::nullopt);
                }

                {
                    BackwardBuilder bb2{"average_pool_backward", gout, gx};
                    if (BackwardBuilder::Target bt2 = bb2.CreateTarget(0)) {
                        bt2.Define([kernel_size, stride, pad, pad_mode](BackwardContext& bctx2) {
                            const Array& ggx = *bctx2.output_grad();
                            bctx2.input_grad() = AveragePool(ggx, kernel_size, stride, pad, pad_mode);
                        });
                    }
                    bb2.Finalize();
                }
                bctx.input_grad() = std::move(gx);
            });
        }
        bb1.Finalize();
    }
    return out;
}

}  // namespace chainerx
