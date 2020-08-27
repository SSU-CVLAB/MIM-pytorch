import argparse
from time import time
import numpy as np
import torch
import os

# 로컬 라이브러리
from layers.SpatioTemporalLSTMCellv2 import SpatioTemporalLSTMCell as STLSTM
from layers.MIMBlock import MIMBlock as MIMblock
from layers.MIMN import MIMN as MIMN

from models.mim import MIM


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
    parser.add_argument('--img_channel', default=1, type=int, help='number of image channel.')
    # model[convlstm, predcnn, predrnn, predrnn_pp]
    parser.add_argument('--model_name', default='convlstm_net', type=str, help='The name of the architecture.')
    parser.add_argument('--pretrained_model', default='', type=str,
                        help='file of a pretrained model to initialize from.')
    parser.add_argument('--num_hidden', default=[64,64,64,64],
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
    parser.add_argument('--batch_size', default=8, type=int, help='batch size for training.')
    parser.add_argument('--max_iterations', default=80000, type=int, help='max num of steps.')
    parser.add_argument('--display_interval', default=1, type=int, help='number of iters showing training loss.')
    parser.add_argument('--test_interval', default=1000, type=int, help='number of iters for test.')
    parser.add_argument('--snapshot_interval', default=1000, type=int, help='number of iters saving models.')
    parser.add_argument('--num_save_samples', default=10, type=int, help='number of sequences to be saved.')
    # gpu
    parser.add_argument('--n_gpu', default=1, type=int, help='how many GPUs to distribute the training across.')
    parser.add_argument('--allow_gpu_growth', default=False, type=bool, help='allow gpu growth')
    parser.add_argument('--img_height', default=0, type=int, help='input image height.')

    args = parser.parse_args()
    return args


def main():
    # 파라미터 로드
    args = parse_args()

    # 리소스 로드
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = MIM(args).to(device)
    print(model)
    print('The model is loaded!!!!\n')

    #    with torch.set_grad_enabled(True):
    #        outputs = model(inputs)


if __name__ == "__main__":
    main()