import os.path
import datetime
import cv2
import numpy as np
from skimage.measure import compare_ssim
from utils import metrics
from utils import preprocess
import torch
import torch.nn.functional as F
import torch.nn as nn

import gc

from torchsummary import summary


def trainer(model, ims, real_input_flag, configs, itr, ims_reverse=None, device=None, optimizer=None, loss=None):
    ims = ims[:, :configs.total_length]
    gt_ims = ims[:, 1:]
    # ims_tmp = np.transpose(ims, (0, 1, 4, 2, 3))
    # ims_list = np.split(ims, configs.n_gpu)
    # ims_list = np.split(ims_tmp, configs.n_gpu)
    # ims_list=np.array(ims_list)  #to np.array
    # ims=np.array(ims)  #to np.array
    # cost = model.forward(ims_list[0], real_input_flag)
    ims_tensor = torch.tensor(ims, device=device)
    gt_ims_tensor = torch.tensor(gt_ims, device=device)

    gen_images = model.forward(ims_tensor, real_input_flag, optimizer)

    loss_value = F.MSELoss(gen_images, gt_ims_tensor)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    loss_print = loss_value.data
    flag = 1

    if configs.reverse_img:
        ims_rev = np.split(ims_reverse, configs.n_gpu)
        gen_images = model.forward(ims_rev, configs.lr, real_input_flag)

        loss_value = loss(gen_images, gt_ims_tensor)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        loss_print += loss_value.data
        flag += 1

    if configs.reverse_input:
        ims_rev = np.split(ims[:, ::-1], configs.n_gpu)
        gen_images = model.forward(ims_rev, configs.lr, real_input_flag)

        loss_value = loss(gen_images, gt_ims_tensor)
        optimizer.zero_grad()
        loss_value.backward(retain_graph=True)
        optimizer.step()

        loss_print += loss_value.data
        flag += 1

        if configs.reverse_img:
            ims_rev = np.split(ims_reverse[:, ::-1], configs.n_gpu)
            gen_images = model.forward(ims_rev, configs.lr, real_input_flag)

            loss_value = loss(gen_images, gt_ims_tensor)
            optimizer.zero_grad()
            loss_value.backward(retain_graph=True)
            optimizer.step()

            loss_print += loss_value.data
            flag += 1

    # _, tensor_loss = cost

    # make_dot(tensor_loss).render('graph', format="png")

    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             print(type(obj), obj.size())
    #     except:
    #         pass

    return loss_print / flag


def test(model, test_input_handle, configs, save_name):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(save_name))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr, fmae, sharp = [], [], [], [], []

    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)

    if configs.img_height > 0:
        height = configs.img_height
    else:
        height = configs.img_width

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - configs.input_length - 1,
         configs.img_width // configs.patch_size,
         height // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    while not test_input_handle.no_batch_left():
        batch_id = batch_id + 1
        if save_name != 'test_result':
            if batch_id > 100: break
        test_ims = test_input_handle.get_batch()
        test_ims = test_ims[:, :configs.total_length]
        if len(test_ims.shape) > 3:
            test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)
        else:
            test_dat = test_ims
        test_dat = np.split(test_dat, configs.n_gpu)
        # 여기서 debug 바꿔줘야 함 현재 im_gen만 나오게 바껴져 있음 원래는 뭐였는지 살펴보기
        with torch.no_grad:
            img_gen, debug = model.test(test_dat, real_input_flag)
            img_gen

        # concat outputs of different gpus along batch
        img_gen = np.concatenate(img_gen)
        if len(img_gen.shape) > 3:
            img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        # MSE per frame
        for i in range(configs.total_length - configs.input_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            x = x[:configs.batch_size * configs.n_gpu]
            x = x - np.where(x > 10000, np.floor_divide(x, 10000) * 10000, np.zeros_like(x))
            gx = img_gen[:, i, :, :, :]
            fmae[i] += metrics.batch_mae_frame_float(gx, x)
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)
            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs.batch_size):
                sharp[i] += np.max(
                    cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))

                score, _ = compare_ssim(gx[b], x[b], full=True, multichannel=True)
                ssim[i] += score

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            if len(debug) != 0:
                np.save(os.path.join(path, "f.npy"), debug)
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                if configs.img_channel == 2:
                    img_gt = img_gt[:, :, :1]
                cv2.imwrite(file_name, img_gt)
            for i in range(configs.total_length - configs.input_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_gen[0, i, :, :, :]
                if configs.img_channel == 2:
                    img_pd = img_pd[:, :, :1]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
        test_input_handle.next()

    avg_mse = avg_mse / (batch_id * configs.batch_size * configs.n_gpu)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size * configs.n_gpu))

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    fmae = np.asarray(fmae, dtype=np.float32) / batch_id
    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    sharp = np.asarray(sharp, dtype=np.float32) / (configs.batch_size * batch_id)

    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])
    print('fmae per frame: ' + str(np.mean(fmae)))
    for i in range(configs.total_length - configs.input_length):
        print(fmae[i])
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])
    print('sharpness per frame: ' + str(np.mean(sharp)))
    for i in range(configs.total_length - configs.input_length):
        print(sharp[i])