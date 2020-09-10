import torch
import torch.nn as nn

import layers.ST_LSTM_def as SLD
import layers.MIMN_def as MND
import layers.MIMBlock_def as MBD

from layers.SpatioTemporalLSTMCellv2 import SpatioTemporalLSTMCell as ST_LSTM
from layers.MIMBlock import MIMBlock as MIM_block
from layers.MIMN import MIMN as MIM_N

from torchviz import make_dot


class MIM(nn.Module):  # ST-LSTM
    def __init__(self, args):
        super(MIM, self).__init__()

        # Tensor shape : [8, 20, 1, 64, 64]
        shape = [args.batch_size, args.total_length, args.patch_size * args.patch_size * args.img_channel,
                 args.img_width // args.patch_size, args.img_height // args.patch_size]

        # 파라미터 로드
        self.num_layers = len(args.num_hidden)  # 4
        self.num_hidden = int(args.num_hidden[1])  # [64, 64, 64, 64]
        self.filter_size = args.filter_size  # 5
        self.total_length = args.total_length  # 20
        self.input_length = args.input_length  # 10
        self.tln = True
        self.device = args.device
        self.batch = args.batch_size
        self._forget_bias = 1.0

        # 모델 파라미터 초기화
        self.gen_images = []
        # self.st_lstm_layer = nn.ModuleList()
        # self.st_lstm_layer_diff = nn.ModuleList()
        self.cell_state = [None, None, None, None]
        self.hidden_state = [None, None, None, None]
        self.cell_state_diff = [None, None, None]
        self.hidden_state_diff = [None, None, None]
        self.conv_lstm_c = None
        self.shape = shape
        self.output_channels = shape[-3]  # 1
        self.width = args.img_width
        self.height = args.img_height

        # ST_LSTM
        num_hidden_in = self.num_hidden
        self.st_t_cc = nn.Conv2d(num_hidden_in, self.num_hidden * 4, self.filter_size, 1, padding=2)
        self.st_s_cc = nn.Conv2d(num_hidden_in, self.num_hidden * 4, self.filter_size, 1, padding=2)
        self.st_x_cc = nn.Conv2d(self.output_channels, self.num_hidden * 4, self.filter_size, 1, padding=2)
        self.st_c_cc = nn.Conv2d(self.num_hidden * 2, self.num_hidden, 1, 1, padding=0)

        self.st_bn_t_cc = nn.BatchNorm2d(self.num_hidden * 4)
        self.st_bn_s_cc = nn.BatchNorm2d(self.num_hidden * 4)
        self.st_bn_x_cc = nn.BatchNorm2d(self.num_hidden * 4)

        # MIM block, MIM S
        # MIM block
        self.mimB_t_cc = nn.Conv2d(num_hidden_in, self.num_hidden * 3, self.filter_size, 1, padding=2)
        self.mimB_s_cc = nn.Conv2d(num_hidden_in, self.num_hidden * 4, self.filter_size, 1, padding=2)
        self.mimB_x_cc = nn.Conv2d(num_hidden_in, self.num_hidden * 4, self.filter_size, 1, padding=2)
        self.mimB_c_cc = nn.Conv2d(self.num_hidden * 2, self.num_hidden, 1, 1, padding=0)

        self.mimB_bn_t_cc = nn.BatchNorm2d(self.num_hidden * 3)
        self.mimB_bn_s_cc = nn.BatchNorm2d(self.num_hidden * 4)
        self.mimB_bn_x_cc = nn.BatchNorm2d(self.num_hidden * 4)

        # MIM S
        self.mimS_h_t = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=2)
        self.mimS_ct_weight = nn.Parameter(torch.randn((self.num_hidden * 2, self.height, self.width)), requires_grad=True)
        self.mimS_x = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=2)
        self.mimS_oc_weight = nn.Parameter(torch.randn((self.num_hidden, self.height, self.width)), requires_grad=True)

        self.mimS_bn_h_concat = nn.BatchNorm2d(self.num_hidden * 4)
        self.mimS_bn_x_concat = nn.BatchNorm2d(self.num_hidden * 4)

        # MIM N
        self.mimN_h_t = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=2)
        self.mimN_ct_weight = nn.Parameter(torch.randn((self.num_hidden * 2, self.height, self.width)), requires_grad=True)
        self.mimN_x = nn.Conv2d(self.num_hidden, self.num_hidden * 4, self.filter_size, 1, padding=2)
        self.mimN_oc_weight = nn.Parameter(torch.randn((self.num_hidden, self.height, self.width)), requires_grad=True)

        self.mimN_bn_h_concat = nn.BatchNorm2d(self.num_hidden * 4)
        self.mimN_bn_x_concat = nn.BatchNorm2d(self.num_hidden * 4)

        # loss func
        # self.loss_fn = nn.MSELoss()

        self.st_lstm = SLD.SpatioTemporalLSTMCell
        self.mimN = MND.MIMN
        self.mimS = MBD.MIMS
        self.mimB = MBD.MIMB
        self.st_memory = None

        # 이미지 생성
        self.x_gen = nn.Conv2d(self.num_hidden, self.output_channels, 1, 1, padding=0)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt' + '-' + str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, args):
        print('load model:', agrs.pretrained_model)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def del_grad(self):
        del self.gen_images
        self.conv_lstm_c = self.conv_lstm_c.clone().detach_()

        for i in range(len(self.hidden_state)):
            self.hidden_state[i] = self.hidden_state[i].clone().detach_()
            self.cell_state[i] = self.cell_state[i].clone().detach_()

        for i in range(len(self.hidden_state_diff)):
            self.hidden_state_diff[i] = self.hidden_state_diff[i].clone().detach_()
            self.cell_state_diff[i] = self.hidden_state_diff[i].clone().detach_()

    # def train(self,args)
    def forward(self, images, schedual_sampling_bool, optimizer):
        self.gen_images = []
        for time_step in range(self.total_length - 1):  # 시간이 길다
            print('time_step: ' + str(time_step))

            if time_step < self.input_length:
                x_gen = images[:, time_step]  # [batch, in_channel,in_height, in_width]
            else:
                # mask
                x_gen = torch.tensor(schedual_sampling_bool[:, time_step - self.input_length],
                                     dtype=torch.double, device=self.device) * images[:, time_step] + \
                        torch.tensor(1 - schedual_sampling_bool[:, time_step - self.input_length],
                                     dtype=torch.double, device=self.device) * x_gen

            preh = self.hidden_state[0]  # 초기화 상태
            # ST_LSTM out: hidden_state[0], cell_state[0], st_memory
            self.hidden_state[0], self.cell_state[0], self.st_memory = self.st_lstm(
                self, x_gen, self.hidden_state[0], self.cell_state[0], self.st_memory)

            for i in range(1, self.num_layers):
                print('i: ' + str(i))
                if time_step > 0:
                    if i == 1:
                        # 먼저 MIM_N 계산
                        self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1] = self.mimN(self,
                            self.hidden_state[i - 1] - preh, self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1])
                    else:
                        self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1] = self.mimN(self,
                            self.hidden_state_diff[i - 2], self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1])
                else:
                    self.mimN(self, torch.zeros_like(self.hidden_state[i - 1]), None, None)

                # MIM_block	계산 마지막hidden_layer state
                preh = self.hidden_state[i]
                self.hidden_state[i], self.cell_state[i], self.st_memory = self.mimB(self,   # MIM_block
                    self.hidden_state[i - 1], self.hidden_state_diff[i - 1], self.hidden_state[i], self.cell_state[i],
                    self.st_memory)

            # 이미지 생성
            x_gen = self.x_gen(self.hidden_state[self.num_layers - 1])
            self.gen_images.append(x_gen)

            # del preh

        self.gen_images = torch.stack(self.gen_images, dim=1)
        # img = images[:, 1:].clone()
        # # img = torch.tensor(images[:, 1:].clone().detach().requires_grad_(True))
        # loss = self.loss_fn(self.gen_images, img)
        # optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        # optimizer.step()

        # self.del_grad()
        # del x_gen

        return self.gen_images
