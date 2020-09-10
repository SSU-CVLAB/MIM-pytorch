import torch
import torch.nn as nn


class MIMBlock(nn.Module):
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
                 seq_shape, x_shape_in, tln=False, device='cpu', initializer=None):
        super(MIMBlock, self).__init__()

        """Initialize the basic Conv LSTM cell.
		Args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			forget_bias: float, The bias added to forget gates (see above).
			tln: whether to apply tensor layer normalization
		"""
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in
        self.device = device

        self.num_hidden = num_hidden
        self.conv_lstm_c = None
        self.batch = seq_shape[0]
        self.height = seq_shape[3]
        self.width = seq_shape[4]
        self.x_shape_in = x_shape_in
        self.layer_norm = tln
        self._forget_bias = 1.0

        # MIM-S
        # h_t
        self.mim_s_h_t = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=2)

        # c_t
        # self.ct_weight = nn.Parameter(torch.randn((self.num_hidden * 2, self.height, self.width)))

        # x
        self.mim_s_x = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=2)

        # oc
        # self.oc_weight = nn.Parameter(torch.randn((self.num_hidden, self.height, self.width)))

        # bn
        self.bn_h_concat = nn.BatchNorm2d(self.num_hidden * 4)
        self.bn_x_concat = nn.BatchNorm2d(self.num_hidden * 4)

        # MIM-BLOCK
        # h
        self.t_cc = nn.Conv2d(self.num_hidden_in, self.num_hidden * 3, self.filter_size, 1, padding=2)

        # m
        self.s_cc = nn.Conv2d(self.num_hidden_in, self.num_hidden * 4, self.filter_size, 1, padding=2)

        # x
        self.x_cc = nn.Conv2d(self.x_shape_in, self.num_hidden * 4, self.filter_size, 1, padding=2)

        # c
        self.c_cc = nn.Conv2d(self.num_hidden * 2, self.num_hidden, 1, 1, padding=0)

        # bn
        self.bn_t_cc = nn.BatchNorm2d(self.num_hidden * 3)
        self.bn_s_cc = nn.BatchNorm2d(self.num_hidden * 4)
        self.bn_x_cc = nn.BatchNorm2d(self.num_hidden * 4)

    # def init_state(self):  # 초기화lstm hidden layer 상태
    #     return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
    #                        dtype=torch.float32, device=self.device, requires_grad=True)

# 내일 weight 빼고 다 del로 넣어버리기
    def MIMS(self, x, h_t, c_t, ct_weight, oc_weight):  # MIMS
        # h_t c_t[batch, in_height, in_width, num_hidden]
        # if h_t is None:
        #     h_t = self.init_state()
        # if c_t is None:
        #     c_t = self.init_state()

        # h_t
        h_concat = self.mim_s_h_t(h_t)

        if self.layer_norm:  # 是否归一化
            h_concat = self.bn_h_concat(h_concat)

        # 셋째차원에서는 4부를 자르기 （ hidden layer 4*num_hidden
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)

        # ct_weight
        ct_activation = torch.mul(c_t.repeat([1, 2, 1, 1]), ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, 1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            # x
            x_concat = self.mim_s_x(x)

            if self.layer_norm:
                x_concat = self.bn_x_concat(x_concat)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, 1)

            i_ += i_x
            f_ += f_x
            g_ += g_x
            o_ += o_x

        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)

        # oc_weight
        o_c = torch.mul(c_new, oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new

    def forward(self, x, diff_h, h, c, m, conv_lstm_c, MIMB_ct_w, MIMB_oc_w):
        # if h is None:
        #     h = self.init_state()
        # if c is None:
        #     c = self.init_state()
        # if m is None:
        #     m = self.init_state()
        # if diff_h is None:
        #     diff_h = torch.zeros_like(h)

        # h
        t_cc = self.t_cc(h)

        # m
        s_cc = self.s_cc(m)

        # x
        x_cc = self.x_cc(x)

        if self.layer_norm:
            t_cc = self.bn_t_cc(t_cc)
            s_cc = self.bn_s_cc(s_cc)
            x_cc = self.bn_x_cc(x_cc)

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
        c, conv_lstm_c = self.MIMS(diff_h, c, conv_lstm_c, MIMB_ct_w, MIMB_oc_w)

        new_c = c + i * g
        cell = torch.cat((new_c, new_m), 1)

        # c
        cell = self.c_cc(cell)

        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m  # [batch, in_height, in_width, num_hidden]
