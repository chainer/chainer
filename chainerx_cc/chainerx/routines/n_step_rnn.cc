#include "chainerx/routines/n_step_rnn.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
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
#include "chainerx/kernel_registry.h"
#include "chainerx/kernels/arithmetic.h"
#include "chainerx/kernels/connection.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/kernels/rnn.h"
#include "chainerx/macro.h"
#include "chainerx/routines/activation.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/hyperbolic.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/misc.h"
#include "chainerx/routines/reduction.h"
#include "chainerx/routines/type_util.h"
namespace chainerx {

namespace {

Array StackWeight(const std::vector<Array>& ws) {
    Array w = Stack(ws, 1);
    StackVector<int64_t, kMaxNdim> shape_vec;
    shape_vec.emplace_back(w.shape()[0] * w.shape()[1]);

    for (int64_t i = 2; i < w.ndim(); i++) {
        shape_vec.emplace_back(w.shape()[i]);
    }

    Shape shape{shape_vec};

    w = Reshape(w, shape);
    return w;
}

std::vector<Array> GruImpl(
        const Array& x,
        const Array& h,
        const absl::optional<Array>& c,
        const std::vector<Array>& ws,
        const std::vector<Array>& bs,
        const absl::optional<std::string>& activation) {
    activation.has_value();
    c.has_value();
    Array xw = Concatenate({ws[0], ws[1], ws[2]}, 0);
    Array hw = Concatenate({ws[3], ws[4], ws[5]}, 0);
    Array xb = Concatenate({bs[0], bs[1], bs[2]}, 0);
    Array hb = Concatenate({bs[3], bs[4], bs[5]}, 0);

    Array gru_x = Linear(x, xw, xb);
    Array gru_h = Linear(h, hw, hb);

    std::vector<Array> split_w = Split(gru_x, 3, 1);
    std::vector<Array> split_h = Split(gru_h, 3, 1);
    Array r_prev = split_w[0] + split_h[0];
    Array r = Sigmoid(r_prev);
    Array z_prev = split_w[1] + split_h[1];
    Array z = Sigmoid(z_prev);
    Array h_bar_prev = split_w[2] + r * split_h[2];
    Array h_bar = Tanh(h_bar_prev);
    std::vector<Array> out{};
    out.reserve(1);
    Array f = (1 - z) * h_bar + z * h;
    out.emplace_back(f);
    return out;
}

std::vector<Array> LstmImpl(
        const Array& x,
        const Array& h,
        const absl::optional<Array>& c,
        const std::vector<Array>& ws,
        const std::vector<Array>& bs,
        const absl::optional<std::string>& activation) {
    activation.has_value();
    std::vector<Array> ws_0_4{ws[2], ws[0], ws[1], ws[3]};
    Array xw = StackWeight(ws_0_4);
    std::vector<Array> ws_5_8{ws[6], ws[4], ws[5], ws[7]};
    Array hw = StackWeight(ws_5_8);
    std::vector<Array> bs_0_4{bs[2], bs[0], bs[1], bs[3]};
    Array xb = StackWeight(bs_0_4);
    std::vector<Array> bs_5_8{bs[6], bs[4], bs[5], bs[7]};
    Array hb = StackWeight(bs_5_8);

    Array lstm_in = Linear(x, xw, xb) + Linear(h, hw, hb);

    std::vector<Array> lstm_out = Lstm(*c, lstm_in);

    return lstm_out;
}

std::vector<Array> RnnImpl(
        const Array& x,
        const Array& h,
        const absl::optional<Array>& c,
        const std::vector<Array>& ws,
        const std::vector<Array>& bs,
        const absl::optional<std::string>& activation) {
    c.has_value();
    Array xw = ws[0];
    Array hw = ws[1];

    Array xb = bs[0];
    Array hb = bs[1];

    Array rnn_in_1 = Linear(x, xw, xb);
    Array rnn_in_2 = Linear(h, hw, hb);
    Array rnn_in = rnn_in_1 + rnn_in_2;
    std::vector<Array> out{};
    out.reserve(1);
    Array rnn_act;
    if (*activation == "tanh") {
        rnn_act = Tanh(rnn_in);
    } else {
        rnn_act = Relu(rnn_in);
    }
    out.emplace_back(rnn_act);
    return out;
}

template <typename Impl>
std::vector<std::vector<Array>> OneDirectionalLoop(
        Impl&& impl,
        std::vector<Array>& xs,
        Array h,
        absl::optional<Array> c,
        const std::vector<Array>& ws,
        const std::vector<Array>& b,
        absl::optional<std::string> activation) {
    Shape h_shape{h.shape()[1], h.shape()[2]};
    h = Reshape(h, h_shape);
    if (c.has_value()) {
        *c = Reshape(*c, h_shape);
    }
    std::vector<Array> h_list;
    for (auto& x : xs) {
        if (x.shape()[0] > h.shape()[0]) {
            throw DimensionError{"The batch size of x must be equal to or less than the size of state", x.shape(), ' ', h.shape()};
        }
        std::vector<int64_t> indices_h;
        indices_h.emplace_back(x.shape()[0]);
        indices_h.emplace_back(h.shape()[0]);
        std::vector<Array> h_split = Split(h, indices_h, 0);
        std::vector<Array> c_split;
        std::vector<Array> h_c;
        if (c.has_value()) {
            std::vector<int64_t> indices_c;
            indices_c.emplace_back(x.shape()[0]);
            indices_c.emplace_back(c->shape()[0]);
            c_split = Split(*c, indices_c, 0);
            h_c = impl(x, h_split[0], c_split[0], ws, b, activation);
        } else {
            h_c = impl(x, h_split[0], c, ws, b, activation);
        }

        h_list.emplace_back(h_c[0]);
        h_split[0] = h_c[0];
        if (c.has_value()) {
            c_split[0] = h_c[1];
            c = Concatenate(c_split, 0);
        }
        h = Concatenate(h_split, 0);
    }
    std::vector<std::vector<Array>> out;
    std::vector<Array> state;
    state.emplace_back(h);
    if (c.has_value()) {
        state.emplace_back(*c);
    }
    out.emplace_back(state);
    out.emplace_back(h_list);

    return out;
}

template <typename Impl>
std::vector<std::vector<Array>> NStepRnnImpl(
        Impl&& impl,
        int64_t n_layers,
        const Array& hx,
        absl::optional<Array> cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        const int8_t use_bidirection,
        const int8_t mode,
        absl::optional<std::string> activation) {
    int8_t direction = use_bidirection ? 2 : 1;
    std::vector<std::vector<Array>> ret;
    if (hx.device().backend().GetName() == "cuda" && hx.dtype() == Dtype::kFloat32) {
        std::vector<std::vector<Array>> out;
        std::shared_ptr<RnnGradState> state{};
        {
            NoBackpropModeScope scope{};
            std::tie(out, state) =
                    hx.device().backend().CallKernel<RnnKernel>(n_layers, hx, cx, ws, bs, xs, use_bidirection, mode, activation);
        }
        {
            std::vector<ConstArrayRef> inp;
            inp.emplace_back(hx);

            if (cx.has_value()) {
                inp.emplace_back(*cx);
            }
            for (int64_t i = 0; i < n_layers; i++) {
                for (int8_t di = 0; di < direction; di++) {
                    for (size_t j = 0; j < ws[0].size(); j++) {
                        int64_t index = direction * i + di;
                        inp.emplace_back(ws[index][j]);
                        inp.emplace_back(bs[index][j]);
                    }
                }
            }

            std::vector<ConstArrayRef> out_grad;
            out_grad.emplace_back(out[0][0]);
            if (cx.has_value()) {
                out_grad.emplace_back(out[0][1]);
            }
            for (size_t i = 0; i < xs.size(); i++) {
                inp.emplace_back(xs[i]);
                out_grad.emplace_back(out[1][i]);
            }
            std::vector<size_t> ind;
            for (size_t i = 0; i < inp.size(); i++) {
                ind.emplace_back(i);
            }
            BackwardBuilder bb{"rnn_backward", std::move(inp), out_grad};

            std::vector<size_t> out_retain;
            size_t start = 1;
            if (cx.has_value()) {
                start = 2;
            }
            for (size_t i = start; i < xs.size() + start; i++) {
                out_retain.emplace_back(i);
            }

            if (BackwardBuilder::Target bt = bb.CreateTarget(ind)) {
                bt.Define([cx_exists = cx.has_value(),
                           state = std::move(state),
                           w_size = ws[0].size(),
                           timesteps = xs.size(),
                           input_toks = bb.RetainInput(ind),
                           out_toks = bb.RetainOutput(out_retain),
                           n_layers,
                           direction,
                           use_bidirection](BackwardContext& bctx) {
                    Array hx = bctx.GetRetainedInput(input_toks[0]);
                    absl::optional<Array> cx = absl::nullopt;
                    int64_t cnt = 1;
                    if (cx_exists) {
                        cx = absl::optional<Array>{bctx.GetRetainedInput(input_toks[1])};
                        cnt = 2;
                    }
                    std::vector<std::vector<Array>> ws;
                    std::vector<std::vector<Array>> bs;

                    for (int64_t i = 0; i < n_layers; i++) {
                        for (int8_t di = 0; di < direction; di++) {
                            std::vector<Array> ws_i;
                            std::vector<Array> bs_i;
                            for (size_t j = 0; j < w_size; j++) {
                                ws_i.emplace_back(bctx.GetRetainedInput(input_toks[cnt++]));
                                bs_i.emplace_back(bctx.GetRetainedInput(input_toks[cnt++]));
                            }
                            ws.emplace_back(ws_i);
                            bs.emplace_back(bs_i);
                        }
                    }

                    std::vector<Array> xs;
                    for (size_t i = cnt; i < cnt + timesteps; i++) {
                        xs.emplace_back(bctx.GetRetainedInput(input_toks[i]));
                    }
                    std::vector<Array> out;
                    for (size_t i = 0; i < timesteps; i++) {
                        out.emplace_back(bctx.GetRetainedOutput(out_toks[i]));
                    }
                    const absl::optional<Array> dhy_n = bctx.output_grad(0);
                    Array dhy;
                    if (dhy_n.has_value()) {
                        dhy = *dhy_n;
                    } else {
                        dhy = Zeros(hx.shape(), hx.dtype(), hx.device());
                    }
                    absl::optional<Array> dcy = absl::nullopt;
                    size_t out_grad_start = 1;
                    if (cx_exists) {
                        out_grad_start = 2;
                        const absl::optional<Array>& dcy_n = bctx.output_grad(1);
                        if (dcy_n.has_value()) {
                            dcy = *dcy_n;
                        } else {
                            dcy = Zeros(hx.shape(), hx.dtype(), hx.device());
                        }
                    }
                    std::vector<Array> dout;
                    for (size_t i = out_grad_start; i < timesteps + out_grad_start; i++) {
                        const absl::optional<Array>& temp_n = bctx.output_grad(i);
                        Array temp;
                        if (temp_n.has_value()) {
                            temp = *temp_n;
                        } else {
                            temp = Zeros({xs[i - out_grad_start].shape()[0], hx.shape()[2] * direction}, hx.dtype(), hx.device());
                        }
                        dout.emplace_back(temp);
                    }

                    std::vector<std::vector<Array>> grad = hx.device().backend().CallKernel<RnnBackwardKernel>(
                            n_layers, hx, cx, ws, bs, xs, dhy, dcy, out, dout, use_bidirection, state);
                    bctx.input_grad(0) = grad[0][0];
                    int64_t grad_ind = 1;

                    if (cx_exists) {
                        grad_ind = 2;
                        bctx.input_grad(1) = grad[0][1];
                    }

                    int64_t initial_grad_ind = grad_ind;
                    for (int64_t i = 0; i < n_layers; i++) {
                        for (int8_t di = 0; di < direction; di++) {
                            for (size_t j = 0; j < w_size; j++) {
                                bctx.input_grad(grad_ind) = grad[1][grad_ind - initial_grad_ind];
                                grad_ind++;
                                bctx.input_grad(grad_ind) = grad[1][grad_ind - initial_grad_ind];
                                grad_ind++;
                            }
                        }
                    }
                    for (size_t i = 0; i < timesteps; i++) {
                        bctx.input_grad(grad_ind) = grad[2][i];
                        grad_ind++;
                    }
                });
            }
            bb.Finalize();
        }
        ret = out;
    } else {
        absl::optional<Array> zero_grad_ = absl::nullopt;
        std::vector<Array> hx_list = Split(hx, hx.shape()[0], 0);
        std::vector<absl::optional<Array>> cx_list;
        if (cx.has_value()) {
            std::vector<Array> cx_list_temp;
            cx_list_temp = Split(*cx, cx->shape()[0], 0);
            for (Array a : cx_list_temp) {
                cx_list.emplace_back(absl::optional<Array>{a});
            }
        } else {
            for (int64_t i = 0; i < hx.shape()[0]; i++) {
                cx_list.emplace_back(zero_grad_);
            }
        }

        std::vector<Array> xs_next = xs;
        std::vector<Array> hy;
        std::vector<Array> cy;
        int64_t idx;
        for (int64_t layer = 0; layer < n_layers; layer++) {
            xs = xs_next;
            idx = direction * layer;
            std::vector<std::vector<Array>> one_directional_out_fw =
                    OneDirectionalLoop(impl, xs, hx_list[idx], cx_list[idx], ws[idx], bs[idx], activation);
            hy.emplace_back(one_directional_out_fw[0][0]);
            if (cx.has_value()) {
                cy.emplace_back(one_directional_out_fw[0][1]);
            }
            std::vector<Array> h_forward = one_directional_out_fw[1];
            if (use_bidirection) {
                idx = direction * layer + 1;
                xs = xs_next;
                std::reverse(xs.begin(), xs.end());

                std::vector<std::vector<Array>> one_directional_out_bw =
                        OneDirectionalLoop(impl, xs, hx_list[idx], cx_list[idx], ws[idx], bs[idx], activation);
                std::reverse(one_directional_out_bw[1].begin(), one_directional_out_bw[1].end());
                std::vector<Array> h_backward = one_directional_out_bw[1];
                xs_next.clear();
                for (size_t i = 0; i < h_backward.size(); i++) {
                    std::vector<Array> h_f_b;
                    h_f_b.emplace_back(h_forward[i]);
                    h_f_b.emplace_back(h_backward[i]);
                    xs_next.emplace_back(Concatenate(h_f_b, 1));
                }
                hy.emplace_back(one_directional_out_bw[0][0]);
                if (cx.has_value()) {
                    cy.emplace_back(one_directional_out_bw[0][1]);
                }
            } else {
                xs_next = h_forward;
            }
        }
        std::vector<Array> ys = xs_next;
        Array h = Stack(hy, 0);
        Array c;
        if (cx.has_value()) {
            c = Stack(cy, 0);
        }
        std::vector<Array> state;
        state.emplace_back(h);
        if (cx.has_value()) {
            state.emplace_back(c);
        }
        std::vector<std::vector<Array>> rnn_out;
        rnn_out.emplace_back(state);
        rnn_out.emplace_back(ys);
        ret = rnn_out;
    }
    return ret;
}
}  // namespace

std::vector<std::vector<Array>> NStepLstm(
        int64_t n_layers,
        const Array& hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs) {
    hx.device().CheckDevicesCompatible(hx, cx, ws[0][0], bs[0][0], xs[0]);
    return NStepRnnImpl(&LstmImpl, n_layers, hx, absl::optional<Array>{cx}, ws, bs, xs, 0, 1, absl::nullopt);
}

std::vector<std::vector<Array>> NStepBiLstm(
        int64_t n_layers,
        const Array& hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs) {
    hx.device().CheckDevicesCompatible(hx, cx, ws[0][0], bs[0][0], xs[0]);
    return NStepRnnImpl(&LstmImpl, n_layers, hx, absl::optional<Array>{cx}, ws, bs, xs, 1, 1, absl::nullopt);
}

std::vector<std::vector<Array>> NStepGru(
        int64_t n_layers,
        const Array& hx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs) {
    hx.device().CheckDevicesCompatible(hx, ws[0][0], bs[0][0], xs[0]);

    return NStepRnnImpl(&GruImpl, n_layers, hx, absl::nullopt, ws, bs, xs, 0, 0, absl::nullopt);
}

std::vector<std::vector<Array>> NStepBiGru(
        int64_t n_layers,
        const Array& hx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs) {
    hx.device().CheckDevicesCompatible(hx, ws[0][0], bs[0][0], xs[0]);

    return NStepRnnImpl(&GruImpl, n_layers, hx, absl::nullopt, ws, bs, xs, 1, 0, absl::nullopt);
}

std::vector<std::vector<Array>> NStepRnn(
        int64_t n_layers,
        const Array& hx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        absl::optional<std::string> activation) {
    hx.device().CheckDevicesCompatible(hx, ws[0][0], bs[0][0], xs[0]);
    return NStepRnnImpl(&RnnImpl, n_layers, hx, absl::nullopt, ws, bs, xs, 0, 2, std::move(activation));
}

std::vector<std::vector<Array>> NStepBiRnn(
        int64_t n_layers,
        const Array& hx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        absl::optional<std::string> activation) {
    hx.device().CheckDevicesCompatible(hx, ws[0][0], bs[0][0], xs[0]);
    return NStepRnnImpl(&RnnImpl, n_layers, hx, absl::nullopt, ws, bs, xs, 1, 2, std::move(activation));
}

}  // namespace chainerx
