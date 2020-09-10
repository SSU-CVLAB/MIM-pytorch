import torch
import torch.nn as nn

def init_state(self):
    return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
                       dtype=torch.float32, device=self.device, requires_grad=False)

def SpatioTemporalLSTMCell(self, x, h, c, m):
    # x [batch, in_channels, in_height, in_width]
    # h c m [batch, num_hidden, in_height, in_width]

    if h is None:
        h = init_state(self)
    if c is None:
        c = init_state(self)
    if m is None:
        m = init_state(self)

    # 네트워크 출력을 계산한다.
    t_cc = self.st_t_cc(h)
    s_cc = self.st_s_cc(m)
    x_gpu = x.clone().to(self.device)
    x_gpu = x_gpu.type(torch.cuda.FloatTensor)
    # x = torch.tensor(x.clone().detach().requires_grad_(True), dtype=torch.float32, device=self.device)
    # x = x.type(torch.cuda.FloatTensor)
    x_cc = self.st_x_cc(x_gpu)

    if self.tln:
        # 평균치,표준편차,norm
        t_cc = self.st_bn_t_cc(t_cc)
        s_cc = self.st_bn_s_cc(s_cc)
        x_cc = self.st_bn_x_cc(x_cc)

    i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, 1)  # [batch, num_hidden, in_height, in_width]
    i_t, g_t, f_t, o_t = torch.split(t_cc, self.num_hidden, 1)
    i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, 1)

    i = torch.sigmoid(i_x + i_t)
    i_ = torch.sigmoid(i_x + i_s)
    g = torch.tanh(g_x + g_t)
    g_ = torch.tanh(g_x + g_s)
    f = torch.sigmoid(f_x + f_t + self._forget_bias)
    f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
    o = torch.sigmoid(o_x + o_t + o_s)
    new_m = f_ * m + i_ * g_
    new_c = f * c + i * g
    cell = torch.cat((new_c, new_m), 1)  # [batch, 2*num_hidden, in_height, in_width]

    cell = self.st_c_cc(cell)
    new_h = o * torch.tanh(cell)

    return new_h, new_c, new_m  # [batch, num_hidden, in_height, in_width]
