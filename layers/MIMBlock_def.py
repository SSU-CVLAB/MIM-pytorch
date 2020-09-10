import torch
import torch.nn as nn


def init_state(self):
    return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
                       dtype=torch.float32, device=self.device, requires_grad=False)


def MIMS(self, x, h_t, c_t):  # MIMS
    # h_t c_t[batch, in_height, in_width, num_hidden]
    if h_t is None:
        h_t = init_state(self)
    if c_t is None:
        c_t = init_state(self)

    # h_t
    h_concat = self.mimS_h_t(h_t)

    if self.tln:  # 是否归一化
        h_concat = self.mimS_bn_h_concat(h_concat)

    # 셋째차원에서는 4부를 자르기 （ hidden layer 4*num_hidden
    i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)

    # ct_weight
    ct_activation = torch.mul(c_t.repeat([1, 2, 1, 1]), self.mimS_ct_weight)
    i_c, f_c = torch.split(ct_activation, self.num_hidden, 1)

    i_ = i_h + i_c
    f_ = f_h + f_c
    g_ = g_h
    o_ = o_h

    if x is not None:
        # x
        x_concat = self.mimS_x(x)

        if self.tln:
            x_concat = self.mimS_bn_x_concat(x_concat)
        i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

        i_ += i_x
        f_ += f_x
        g_ += g_x
        o_ += o_x

    i_ = torch.sigmoid(i_)
    f_ = torch.sigmoid(f_ + self._forget_bias)
    c_new = f_ * c_t + i_ * torch.tanh(g_)

    # oc_weight
    o_c = torch.mul(c_new, self.mimS_oc_weight)

    h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

    return h_new, c_new


def MIMB(self, x, diff_h, h, c, m):
    if h is None:
        h = init_state(self)
    if c is None:
        c = init_state(self)
    if m is None:
        m = init_state(self)
    if diff_h is None:
        diff_h = torch.zeros_like(h)

    # h
    t_cc = self.mimB_t_cc(h)

    # m
    s_cc = self.mimB_s_cc(m)

    # x
    x_cc = self.mimB_x_cc(x)

    if self.tln:
        t_cc = self.mimB_bn_t_cc(t_cc)
        s_cc = self.mimB_bn_s_cc(s_cc)
        x_cc = self.mimB_bn_x_cc(x_cc)

    i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, 1)
    i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, 1)
    i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, 1)

    i = torch.sigmoid(i_x + i_t)
    i_ = torch.sigmoid(i_x + i_s)
    g = torch.tanh(g_x + g_t)
    g_ = torch.tanh(g_x + g_s)
    f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
    o = torch.sigmoid(o_x + o_t + o_s)
    new_m = f_ * m + i_ * g_

    # MIMS
    c, self.conv_lstm_c = self.mimS(self, diff_h, c, self.conv_lstm_c)

    new_c = c + i * g
    cell = torch.cat((new_c, new_m), 1)

    # c
    cell = self.mimB_c_cc(cell)

    new_h = o * torch.tanh(cell)

    return new_h, new_c, new_m  # [batch, in_height, in_width, num_hidden]