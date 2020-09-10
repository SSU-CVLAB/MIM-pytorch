import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):  # ST-LSTM
    def __init__(self, layer_name, filter_size, num_hidden_in, num_hidden,
                 seq_shape, x_shape_in, tln=False, device='cpu', initializer=None):
        super(SpatioTemporalLSTMCell, self).__init__()

        '''
        Initialize the basic Conv LSTM cell.
		args:
			layer_name: layer names for different convlstm layers.
			filter_size: int tuple thats the height and width of the filter.
			num_hidden: number of units in output tensor.
			forget_bias: float, The bias added to forget gates (see above).
			tln: whether to apply tensor layer normalization
		'''

        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_hidden_in = num_hidden_in  # TypeError: not all arguments converted during string formatting
        self.num_hidden = num_hidden  # TypeError: not all arguments converted during string formatting
        self.batch = seq_shape[0]
        self.height = seq_shape[3]
        self.width = seq_shape[4]
        self.x_shape_in = x_shape_in
        self.layer_norm = tln
        self._forget_bias = 1.0
        self.device = device

        # 레이어 정의
        # h
        self.t_cc = nn.Conv2d(self.num_hidden_in,
                              self.num_hidden * 4,
                              self.filter_size, 1, padding=2
                              )

        # m
        self.s_cc = nn.Conv2d(self.num_hidden_in,
                              self.num_hidden * 4,
                              self.filter_size, 1, padding=2
                              )

        # x
        self.x_cc = nn.Conv2d(self.x_shape_in,
                              self.num_hidden * 4,
                              self.filter_size, 1, padding=2
                              )

        # c
        self.c_cc = nn.Conv2d(self.num_hidden * 2,
                              self.num_hidden,
                              1, 1, padding=0
                              )

        # bn
        self.bn_t_cc = nn.BatchNorm2d(self.num_hidden * 4)
        self.bn_s_cc = nn.BatchNorm2d(self.num_hidden * 4)
        self.bn_x_cc = nn.BatchNorm2d(self.num_hidden * 4)

    # def init_state(self):
    #     return torch.zeros((self.batch, self.num_hidden, self.height, self.width),
    #                        dtype=torch.float32, device=self.device, requires_grad=True)

    def forward(self, x, h, c, m):
        # x [batch, in_channels, in_height, in_width]
        # h c m [batch, num_hidden, in_height, in_width]

        # if h is None:
        #     h = self.init_state()
        # if c is None:
        #     c = self.init_state()
        # if m is None:
        #     m = self.init_state()

        # 네트워크 출력을 계산한다.
        t_cc = self.t_cc(h)
        s_cc = self.s_cc(m)
        x = x.type(torch.cuda.FloatTensor)
        # x = torch.tensor(x.clone(), dtype=torch.float32, device=self.device)
        x_cc = self.x_cc(x)

        if self.layer_norm:
            # 평균치,표준편차,norm
            t_cc = self.bn_t_cc(t_cc)
            s_cc = self.bn_s_cc(s_cc)
            x_cc = self.bn_x_cc(x_cc)

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

        cell = self.c_cc(cell)
        new_h = o * torch.tanh(cell)

        return new_h, new_c, new_m  # [batch, num_hidden, in_height, in_width]
