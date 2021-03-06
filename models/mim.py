import torch
import torch.nn as nn

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
        self.num_hidden = args.num_hidden  # [64, 64, 64, 64]
        self.filter_size = args.filter_size  # 5
        self.total_length = args.total_length  # 20
        self.input_length = args.input_length  # 10
        self.tln = True
        self.device = args.device
        self.batch = args.batch_size
        self._forget_bias = 1.0

        # 모델 파라미터 초기화
        self.gen_images = []
        self.st_lstm_layer = nn.ModuleList()
        self.st_lstm_layer_diff = nn.ModuleList()
        self.cell_state = []
        self.hidden_state = []
        self.cell_state_diff = []
        self.hidden_state_diff = []
        self.shape = shape
        self.output_channels = shape[-3]  # 1

        for i in range(self.num_layers):
            if i == 0:
                num_hidden_in = int(self.num_hidden[self.num_layers - 1])  # hidden input 64
            else:
                num_hidden_in = int(self.num_hidden[i - 1])  # hidden input

            if i < 1:  # ST-LSTM
                new_st_lstm_layer = ST_LSTM('ST_LSTM_' + str(i + 1),
                                            self.filter_size,
                                            num_hidden_in,
                                            int(self.num_hidden[i]),
                                            self.shape,
                                            self.output_channels,  # 64
                                            tln=self.tln,
                                            device=self.device)
            else:  # MIM-block
                new_st_lstm_layer = MIM_block('ST_LSTM_' + str(i + 1),
                                              self.filter_size,
                                              num_hidden_in,
                                              self.num_hidden[i],
                                              self.shape,
                                              self.num_hidden[i - 1],
                                              tln=self.tln,
                                              device=self.device)
            self.st_lstm_layer.append(new_st_lstm_layer)
            self.cell_state.append(None)  # 메모리
            self.hidden_state.append(None)  # state

        for i in range(self.num_layers - 1):  # 추가 MIMN
            new_st_lstm_layer = MIM_N('ST_LSTM_diff' + str(i + 1),
                                      self.filter_size,
                                      self.num_hidden[i + 1],
                                      self.shape,
                                      tln=self.tln,
                                      device=self.device)
            self.st_lstm_layer_diff.append(new_st_lstm_layer)
            self.cell_state_diff.append(None)  # 메모리
            self.hidden_state_diff.append(None)  # state

        self.st_memory = None

        # 이미지 생성
        self.x_gen = nn.Conv2d(self.num_hidden[self.num_layers - 1],
                               self.output_channels, 1, 1, padding=0
                               )

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

    # def train(self,args)
    def forward(self, images, schedual_sampling_bool):
        self.gen_images = []
        for time_step in range(self.total_length - 1):  # 시간이 길다
            # print('time_step: ' + str(time_step))

            if time_step < self.input_length:  #
                x_gen = images[:, time_step]  # [batch, in_channel,in_height, in_width]
            else:
                # mask
                x_gen = torch.tensor(schedual_sampling_bool[:, time_step - self.input_length],
                                     dtype=torch.double, device=self.device) * images[:, time_step] + \
                        torch.tensor(1 - schedual_sampling_bool[:, time_step - self.input_length],
                                     dtype=torch.double, device=self.device) * x_gen

            preh = self.hidden_state[0]  # 초기화 상태
            self.hidden_state[0], self.cell_state[0], self.st_memory = self.st_lstm_layer[0](
                # ST_LSTM out: hidden_state[0], cell_state[0], st_memory
                x_gen, self.hidden_state[0], self.cell_state[0], self.st_memory)

            # MIM_block
            for i in range(1, self.num_layers):
                # print('i: ' + str(i))
                if time_step > 0:
                    if i == 1:
                        # make_dot(self.st_lstm_layer_diff[i - 1](
                        #         self.hidden_state[i - 1] - preh,
                        #         self.hidden_state_diff[i - 1],
                        #         self.cell_state_diff[i - 1])).render('mimn', format='png')

                        # 먼저 MIM_N 계산
                        self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1] = self.st_lstm_layer_diff[i - 1](
                            self.hidden_state[i - 1] - preh, self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1])
                    else:
                        self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1] = self.st_lstm_layer_diff[i - 1](
                            # 먼저 MIM_N 계산
                            self.hidden_state_diff[i - 2], self.hidden_state_diff[i - 1], self.cell_state_diff[i - 1])
                else:
                    self.st_lstm_layer_diff[i - 1](torch.zeros_like(self.hidden_state[i - 1]), None, None)

                # MIM_block	계산 마지막hidden_layer state
                preh = self.hidden_state[i]

                # make_dot(self.st_lstm_layer[i](  # MIM_block
                #     self.hidden_state[i - 1], self.hidden_state_diff[i - 1], self.hidden_state[i], self.cell_state[i],
                #     self.st_memory)).render('mim block', format='png')

                self.hidden_state[i], self.cell_state[i], self.st_memory = self.st_lstm_layer[i](  # MIM_block
                    self.hidden_state[i - 1], self.hidden_state_diff[i - 1], self.hidden_state[i], self.cell_state[i],
                    self.st_memory)

            # 이미지 생성
            x_gen = self.x_gen(self.hidden_state[self.num_layers - 1])

            self.gen_images.append(x_gen)

        self.gen_images = torch.stack(self.gen_images, dim=1)
        img = torch.tensor(images[:, 1:])
        loss = self.loss_fn(self.gen_images, img)

        return self.gen_images, loss
