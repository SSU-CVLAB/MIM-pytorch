import argparse
from time import time
import numpy as np
import torch
import datetime
import os

# 로컬 라이브러리
from layers.SpatioTemporalLSTMCellv2 import SpatioTemporalLSTMCell as STLSTM
from layers.MIMBlock import MIMBlock as MIMblock
from layers.MIMN import MIMN as MIMN

from data_provider import datasets_factory
from utils import preprocess
from models.mim import MIM
# from models.mim2 import MIM

import trainer


def parse_args():
    parser = argparse.ArgumentParser(description="tensorflow to pytorch for MIM")
    # mode
    parser.add_argument("--is_training", default=True, type=bool, help="training or testing")
    # data I/O
    parser.add_argument("--dataset_name", default="mnist", type=str, help="The name of dataset")
    parser.add_argument("--train_data_paths", default="data/moving-mnist-example/moving-mnist-train.npz", type=str,
                        help="train data paths")
    parser.add_argument('--valid_data_paths', default='data/moving-mnist-example/moving-mnist-valid.npz', type=str,
                        help='validation data paths.')
    parser.add_argument('--save_dir', default='checkpoints/mnist_MIM_pp', type=str,
                        help='dir to store trained net.')
    parser.add_argument('--gen_frm_dir', default='results/mnist_predrnn_pp', type=str, help='dir to store result.')
    parser.add_argument('--input_length', default=5, type=int, help='encoder hidden states.')
    parser.add_argument('--total_length', default=10, type=int, help='total input and output length.')
    parser.add_argument('--img_width', default=64, type=int, help='input image width.')
    parser.add_argument('--img_height', default=64, type=int, help='input image height.')
    parser.add_argument('--img_channel', default=1, type=int, help='number of image channel.')
    # model[convlstm, predcnn, predrnn, predrnn_pp]
    parser.add_argument('--model_name', default='convlstm_net', type=str, help='The name of the architecture.')
    parser.add_argument('--pretrained_model', default='', type=str,
                        help='file of a pretrained model to initialize from.')
    parser.add_argument('--num_hidden', default=[64, 64, 64, 64],
                        help='COMMA separated number of units in a convlstm layer.')
    parser.add_argument('--filter_size', default=5, type=int, help='filter of a convlstm layer.')
    parser.add_argument('--stride', default=1, type=int, help='stride of a convlstm layer.')
    parser.add_argument('--patch_size', default=1, type=int, help='patch size on one dimension.')
    parser.add_argument('--layer_norm', default=True, type=bool, help='whether to apply tensor layer norm.')
    # scheduled sampling
    parser.add_argument('--scheduled_sampling', default=True, type=bool, help='for scheduled sampling')
    parser.add_argument('--sampling_stop_iter', default=50000, type=int, help='for scheduled sampling.')
    parser.add_argument('--sampling_start_value', default=1.0, type=float, help='for scheduled sampling.')
    parser.add_argument('--sampling_changing_rate', default=0.00002, type=float, help='for scheduled sampling.')
    # optimization
    parser.add_argument('--lr', default=0.001, type=float, help='base learning rate.')
    parser.add_argument('--reverse_input', default=False, type=bool,
                        help='whether to reverse the input frames while training.')
    parser.add_argument('--reverse_img', default=False, type=bool,
                        help='whether to reverse the input images while training.')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for training.')
    parser.add_argument('--max_iterations', default=80000, type=int, help='max num of steps.')
    parser.add_argument('--display_interval', default=1, type=int, help='number of iters showing training loss.')
    parser.add_argument('--test_interval', default=1000, type=int, help='number of iters for test.')
    parser.add_argument('--snapshot_interval', default=1000, type=int, help='number of iters saving models.')
    parser.add_argument('--num_save_samples', default=10, type=int, help='number of sequences to be saved.')
    # gpu
    parser.add_argument('--n_gpu', default=1, type=int, help='how many GPUs to distribute the training across.')
    parser.add_argument('--allow_gpu_growth', default=False, type=bool, help='allow gpu growth')
    parser.add_argument('--device', default='cuda', type=str, help='Training device')

    args = parser.parse_args()
    return args


def schedule_sampling(eta, itr, args):
    if args.img_height > 0:
        height = args.img_height
    else:
        height = args.img_width
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.patch_size ** 2 * args.img_channel,
                      args.img_width // args.patch_size,
                      height // args.patch_size))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:  # 50000
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0

    random_flip = np.random.random_sample((args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.patch_size ** 2 * args.img_channel,
                    args.img_width // args.patch_size,
                    height // args.patch_size))
    zeros = np.zeros((args.patch_size ** 2 * args.img_channel,
                      args.img_width // args.patch_size,
                      height // args.patch_size))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)

    # real_input_flag is list. so we should change to numpy array for reshaping
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.patch_size ** 2 * args.img_channel,
                                  args.img_width // args.patch_size,
                                  height // args.patch_size))
    return eta, real_input_flag


def main():
    # 파라미터 로드
    args = parse_args()

    # 리소스 로드
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")
    model = MIM(args).to(device)
    print(model)
    print('The model is loaded!\n')

    # 데이터셋 로드
    train_input_handle, test_input_handle = datasets_factory.data_provider(args.dataset_name,
                                                                           args.train_data_paths,
                                                                           args.valid_data_paths,
                                                                           args.batch_size * args.n_gpu,
                                                                           args.img_width,
                                                                           seq_length=args.total_length,
                                                                           is_training=True)  # n 64 64 1 로 나옴

    # with torch.set_grad_enabled(True):
    if args.pretrained_model:
        model.load(args.pretrained_model)

    eta = args.sampling_start_value  # 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    MSELoss = torch.nn.MSELoss()

    for itr in range(1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)

        ims = train_input_handle.get_batch()
        ims_reverse = None
        if args.reverse_img:
            ims_reverse = ims[:, :, :, ::-1]
            ims_reverse = preprocess.reshape_patch(ims_reverse, args.patch_size)
        ims = preprocess.reshape_patch(ims, args.patch_size)
        eta, real_input_flag = schedule_sampling(eta, itr, args)

        loss = trainer.trainer(model, ims, real_input_flag, args, itr, ims_reverse, device, optimizer, MSELoss)

        if itr % args.snapshot_interval == 0:
            model.save(itr)

        if itr % args.test_interval == 0:
            trainer.test(model, test_input_handle, args, itr)

        if itr % args.display_interval == 0:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
            print('training loss: ' + str(loss))

        train_input_handle.next()

        del loss


if __name__ == "__main__":
    main()
