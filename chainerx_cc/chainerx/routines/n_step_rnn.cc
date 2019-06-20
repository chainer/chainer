#include "chainerx/routines/n_step_rnn.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <nonstd/optional.hpp>

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
#include "chainerx/kernels/connection.h"
#include "chainerx/kernels/rnn.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/kernels/math.h"
#include "chainerx/macro.h"
#include "chainerx/routines/connection.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/stack_vector.h"

namespace chainerx {

Array _stack_weight(const std::vector<Array>& ws) {
    Array w = Stack(ws, 1);
    StackVector<int64_t, kMaxNdim> shape_vec;
    shape_vec.push_back(w.shape()[0] * w.shape()[1]);

    for (int64_t i = 2; i < w.ndim(); i++) {
        shape_vec.push_back(w.shape()[i]);
    }

    Shape shape{shape_vec};

    w = Reshape(w, shape);
    return w;
}

std::vector<Array> _lstm(
        const Array& x, const Array& h, const nonstd::optional<Array>& c, const std::vector<Array>& ws, const std::vector<Array>& bs) {
    std::vector<Array> ws_0_4{ws[2], ws[0], ws[1], ws[3]};
    Array xw = _stack_weight(ws_0_4);
    std::vector<Array> ws_5_8{ws[6], ws[4], ws[5], ws[7]};
    Array hw = _stack_weight(ws_5_8);
    std::vector<Array> bs_0_4{bs[2], bs[0], bs[1], bs[3]};
    Array xb = _stack_weight(bs_0_4);
    std::vector<Array> bs_5_8{bs[6], bs[4], bs[5], bs[7]};
    Array hb = _stack_weight(bs_5_8);

    Array lstm_in = Linear(x, xw, xb) + Linear(h, hw, hb);

    std::vector<Array> lstm_out = lstm(*c, lstm_in);

    return lstm_out;
}

template <typename Impl>
std::vector<std::vector<Array>> _one_directional_loop(
        Impl&& impl,
        std::vector<Array>& xs,
        Array h,
        nonstd::optional<Array> c,
        const std::vector<Array>& ws,
        const std::vector<Array>& b) {
    Shape h_shape{h.shape()[1], h.shape()[2]};
    h = Reshape(h, h_shape);
    if (c.has_value()) {
        *c = Reshape(*c, h_shape);
    }
    std::vector<Array> h_list;
    for (uint i = 0; i < xs.size(); i++) {
        Array x_t = xs[i];

        if (x_t.shape()[0] > h.shape()[0]) {
            throw DimensionError{"The batch size of x must be equal to or less than the size of state", x_t.shape(), ' ', h.shape()};
        }
        std::vector<int64_t> indices_h;
        indices_h.push_back(x_t.shape()[0]);
        indices_h.push_back(h.shape()[0]);
        std::vector<Array> h_split = Split(h, indices_h, 0);
        std::vector<Array> c_split;
        std::vector<Array> h_c;

        if (c.has_value()) {
            std::vector<int64_t> indices_c;
            indices_c.push_back(x_t.shape()[0]);
            indices_c.push_back(c->shape()[0]);
            c_split = Split(*c, indices_c, 0);
            h_c = impl(xs[i], h_split[0], c_split[0], ws, b);
        } else {
            h_c = impl(xs[i], h_split[0], nonstd::nullopt, ws, b);
        }

        h_list.push_back(h_c[1]);
        h_split[0] = h_c[1];
        if (c.has_value()) {
            c_split[0] = h_c[0];
            c = Concatenate(c_split, 0);
        }
        h = Concatenate(h_split, 0);
    }
    std::vector<std::vector<Array>> out;
    std::vector<Array> state;
    state.push_back(h);
    state.push_back(*c);
    out.push_back(state);
    out.push_back(h_list);

    return out;
}

template <typename Impl>
std::vector<std::vector<Array>> n_step_rnn_impl(
        Impl&& impl,
        int64_t n_layers,
        Array hx,
        nonstd::optional<Array> cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs,
        const int8_t use_bidirection) {
    int8_t direction = use_bidirection ? 2 : 1;
    std::vector<Array> hx_list = Split(hx, hx.shape()[0], 0);
    std::vector<Array> cx_list;
    if (cx.has_value()) {
        cx_list = Split(*cx, cx->shape()[0], 0);
    } else {
        for (int64_t i = 0; i < hx.shape()[0]; i++) {
            cx_list.push_back((Array)NULL);
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
                _one_directional_loop(impl, xs, hx_list[idx], cx_list[idx], ws[idx], bs[idx]);
        hy.push_back(one_directional_out_fw[0][0]);
        cy.push_back(one_directional_out_fw[0][1]);
        std::vector<Array> h_forward = one_directional_out_fw[1];
        if (use_bidirection) {
            idx = direction * layer + 1;
            xs = xs_next;
            std::reverse(xs.begin(), xs.end());
            std::vector<std::vector<Array>> one_directional_out_bw =
                    _one_directional_loop(impl, xs, hx_list[idx], cx_list[idx], ws[idx], bs[idx]);
            std::reverse(one_directional_out_bw[1].begin(), one_directional_out_bw[1].end());
            std::vector<Array> h_backward = one_directional_out_bw[1];
            xs_next.clear();
            for (uint i = 0; i < h_backward.size(); i++) {
                std::vector<Array> h_f_b;
                h_f_b.push_back(h_forward[i]);
                h_f_b.push_back(h_backward[i]);
                xs_next.push_back(Concatenate(h_f_b, 1));
            }
            hy.push_back(one_directional_out_bw[0][0]);
            cy.push_back(one_directional_out_bw[0][1]);
        } else {
            xs_next = h_forward;
        }
    }
    std::vector<Array> ys = xs_next;
    Array h = Stack(hy, 0);
    Array c;
    if (cx.has_value()) {
        c = Stack(cy, 0);
    } else {
        c = (Array)NULL;
    }
    std::vector<Array> state;
    state.push_back(h);
    state.push_back(c);
    std::vector<std::vector<Array>> rnn_out;
    rnn_out.push_back(state);
    rnn_out.push_back(ys);
    return rnn_out;
}

std::vector<std::vector<Array>> n_step_lstm(
        int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs) {
    // assuming that all arrays in a list should belong to the same device
    hx.device().CheckDevicesCompatible(hx, cx, ws[0][0], bs[0][0], xs[0]); 
    if(hx.device().backend().GetName() == "cuda")
    {
        std::vector<std::vector<Array>> out;
        {
            NoBackpropModeScope scope{};
            out = hx.device().backend().CallKernel<RnnKernel>(n_layers, hx, cx, ws, bs, xs, 0, 1);
        }
        {
            std::vector<ConstArrayRef> inp;
            inp.push_back(hx);
            inp.push_back(cx);
            for(int64_t i = 0; i < n_layers; i++) {
                for(int64_t j = 0; j < 8; j++) {
                    inp.push_back(ws[i][j]);
                    inp.push_back(bs[i][j]);
                }
            }
            std::vector<ConstArrayRef> out_grad;
            out_grad.push_back(out[0][0]);
            out_grad.push_back(out[0][1]); 
            for(uint i = 0; i < xs.size(); i++) {
                inp.push_back(xs[i]);
                out_grad.push_back(out[1][i]);
            }


            BackwardBuilder bb{"lstm_backward", inp, out_grad};

            std::vector<size_t> ind;
            for(uint i = 0; i < inp.size(); i++) {
                ind.push_back(i);
            }
            if(BackwardBuilder::Target bt = bb.CreateTarget(ind)) {
                
                std::vector<size_t> out_retain;
                for(uint i = 2; i < xs.size() + 2; i++) {
                    out_retain.push_back(i);
                }
                bt.Define([timesteps = xs.size(), input_toks = bb.RetainInput(ind), out_toks = bb.RetainOutput(out_retain), n_layers](BackwardContext& bctx) {
                    Array hx = bctx.GetRetainedInput(input_toks[0]);
                    Array cx = bctx.GetRetainedInput(input_toks[1]);
                    std::vector<std::vector<Array>> ws;
                    std::vector<std::vector<Array>> bs;
                    int cnt = 2;
                    for(int i = 0; i < n_layers; i++) {
                        std::vector<Array> ws_i;
                        std::vector<Array> bs_i;
                        for(int j = 0; j < 8; j++) {
                            ws_i.push_back(bctx.GetRetainedInput(input_toks[cnt++]));
                            bs_i.push_back(bctx.GetRetainedInput(input_toks[cnt++]));
                        }
                        ws.push_back(ws_i);
                        bs.push_back(bs_i);
                    }
                    std::vector<Array> xs;
                    for(uint i = cnt; i < cnt + timesteps; i++) {
                        xs.push_back(bctx.GetRetainedInput(input_toks[i]));
                    }

                    std::vector<Array> out;
                    for(uint i = 0; i < timesteps; i++) {
                        out.push_back(bctx.GetRetainedOutput(out_toks[i]));
                    }
                    const nonstd::optional<Array> dhy_n = bctx.output_grad(0);
                    Array dhy;
                    if(dhy_n.has_value()) {

                        dhy = *dhy_n;
                    } else{

                        dhy = Zeros(hx.shape(), hx.dtype(), hx.device());
                    }
                    const nonstd::optional<Array> dcy_n = bctx.output_grad(1);
                    Array dcy;
                    if(dcy_n.has_value()) {
                        dcy = *dcy_n;
                    } else {
                        dcy = Zeros(cx.shape(), cx.dtype(), cx.device());
                    }
                    std::vector<Array> dout;


                    for(uint i = 2; i < xs.size()+2; i++) {

                        const nonstd::optional<Array> temp_n = bctx.output_grad(i);
                        Array temp;
                        if(temp_n.has_value()) {
                            temp = *temp_n;
                        } else {
                            temp = Zeros({xs[i-2].shape()[0], hx.shape()[2]}, hx.dtype(), hx.device());
                        }
                        dout.push_back(temp);
                    }
                    std::vector<std::vector<Array>> grad = hx.device().backend().CallKernel<RnnBackwardKernel>((int)n_layers, hx, cx, ws, bs, xs, dhy, dcy, out, dout, 0, 1);
                    bctx.input_grad(0) = grad[0][0];
                    bctx.input_grad(1) = grad[0][1];
                    int grad_ind = 2;
                    for(int64_t i = 0; i <  n_layers; i++) {
                        for(int64_t j = 0; j <  8; j++) {
                            bctx.input_grad(grad_ind) = grad[2][grad_ind - 2];
                            grad_ind++;
                            bctx.input_grad(grad_ind) = grad[2][grad_ind - 2];
                            grad_ind++;
                        }
                    }

                    for(uint i = 0; i < xs.size(); i++) {

                        bctx.input_grad(grad_ind++) = grad[1][i];
                    }
                });
            }
        }
        return out;
    } else {
    return n_step_rnn_impl(&_lstm, n_layers, hx, nonstd::optional<Array>{cx}, ws, bs, xs, 0);
    }
}

std::vector<std::vector<Array>> n_step_bilstm(
        int64_t n_layers,
        Array hx,
        Array cx,
        const std::vector<std::vector<Array>>& ws,
        const std::vector<std::vector<Array>>& bs,
        std::vector<Array>& xs) {
    
    return n_step_rnn_impl(&_lstm, n_layers, hx, nonstd::optional<Array>{cx}, ws, bs, xs, 1);
    
}


}  // namespace chainerx
