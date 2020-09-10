import torch
import torch.nn as nn


class MIMN(nn.Module):
    def __init__(self, layer_name, filter_size, num_hidden, seq_shape, tln=True, device='cpu', initializer=0.001):
        super(MIMN, self).__init__()
        """Initialize the basic Conv LSTM cell.
		Args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			tln: whether to apply tensor layer normalization.
		"""
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden = num_hidden
        self.layer_norm = tln
        self.batch = seq_shape[0]
        self.height = seq_shape[3]
        self.width = seq_shape[4]
        self._forget_bias = 1.0
        self.device = device

        # h_t
        self.h_t = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=2)

        # c_t
        # self.ct_weight = nn.Parameter(torch.randn((self.num_hidden * 2, self.height, self.width)))

        # x
        self.x = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=2)

        # oc
        # self.oc_weight = nn.Parameter(torch.randn((self.num_hidden, self.height, self.width)))

        # bn
        self.bn_h_concat = nn.BatchNorm2d(self.num_hidden * 4)
        self.bn_x_concat = nn.BatchNorm2d(self.num_hidden * 4)

    # def init_state(self):
    #     shape = [self.batch, self.num_hidden, self.height, self.width]
    #     return torch.zeros(shape, dtype=torch.float32, device=self.device, requires_grad=True)

    def forward(self, x, h_t, c_t, ct_weight, oc_weight):

        # h c [batch, num_hidden, in_height, in_width]
        # if h_t is None:
        #     h_t = self.init_state()
        # if c_t is None:
        #     c_t = self.init_state()

        # 1
        h_concat = self.h_t(h_t)

        if self.layer_norm:
            h_concat = self.bn_h_concat(h_concat)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, 1)

        ct_activation = torch.mul(c_t.repeat([1, 2, 1, 1]), ct_weight)
        i_c, f_c = torch.split(ct_activation, self.num_hidden, 1)

        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h

        if x is not None:
            # 3 x
            x_concat = self.x(x)

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

        o_c = torch.mul(c_new, oc_weight)

        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)

        return h_new, c_new  # [batch, in_height, in_width, num_hidden]
