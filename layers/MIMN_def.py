import torch
import torch.nn as nn


def init_state(self):
    shape = [self.batch, self.num_hidden, self.height, self.width]
    return torch.zeros(shape, dtype=torch.float32, device=self.device, requires_grad=False)


def MIMN(self, x, h_t, c_t):
    # h c [batch, num_hidden, in_height, in_width]
    if h_t is None:
        h_t = init_state(self)
    if c_t is None:
        c_t = init_state(self)

    # 1
    h_concat = self.mimN_h_t(h_t)

    if self.tln:
        h_concat = self.mimN_bn_h_concat(h_concat)
    i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)

    ct_activation = torch.mul(c_t.repeat([1, 2, 1, 1]), self.mimN_ct_weight)
    i_c, f_c = torch.split(ct_activation, self.num_hidden, 1)

    i_ = i_h + i_c
    f_ = f_h + f_c
    g_ = g_h
    o_ = o_h

    if x is not None:
        # 3 x
        x_concat = self.mimN_x(x)

        if self.tln:
            x_concat = self.mimN_bn_x_concat(x_concat)
        i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

        i_ += i_x
        f_ += f_x
        g_ += g_x
        o_ += o_x

    i_ = torch.sigmoid(i_)
    f_ = torch.sigmoid(f_ + self._forget_bias)
    c_new = f_ * c_t + i_ * torch.tanh(g_)

    o_c = torch.mul(c_new, self.mimN_oc_weight)

    h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

    return h_new, c_new  # [batch, in_height, in_width, num_hidden]