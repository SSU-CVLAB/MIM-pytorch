import cv2
import numpy as np
import torch


def calc_optical_flow_loss(gen_diff, gt_diff, device='cpu'):
    gen_diff_tensor = torch.tensor(gen_diff, device=device, requires_grad=True)
    gt_diff_tensor = torch.tensor(gt_diff, device=device, requires_grad=True)

    # optical flow loss 벡터 구하는 식
    diff = gt_diff_tensor - gen_diff_tensor
    diff = torch.pow(diff, 2)
    squared_distance = diff[0] + diff[1]
    distance = torch.sqrt(squared_distance)
    distance_sum = torch.mean(distance)

    return distance_sum


def to_grayscale(images):
    binary_images = []

    for seq in images:
        for img in seq:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary_images.append(img.copy())

    return binary_images


def gpu_to_cpu(images):
    cpu_images = []

    for seq in images:
        seq_images = []

        for img in seq:
            seq_images.append(img.clone().detach().to('cpu').numpy().reshape((img.shape[1], img.shape[2], img.shape[0])))

        cpu_images.append(seq_images.copy())

    return cpu_images


def float_to_cv8u(array):
    min = np.min(array)
    array = array - min
    max = np.max(array)
    div = max / float(255)
    array = np.uint8(np.round(array / div))

    return array


def normalize(array):
    return array / 127.5 - 1


def dense_optical_flow_loss(gen_images, gt_images, img_channel):
    optical = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    gen_images_cpu = gpu_to_cpu(gen_images)
    gt_images_cpu = gpu_to_cpu(gt_images)

    if img_channel == 3:
        gen_images_cpu = to_grayscale(gen_images_cpu)
        gt_images_cpu = to_grayscale(gt_images_cpu)

    gen_diff = []
    gt_diff = []
    for gen_seq, gt_seq in zip(gen_images_cpu, gt_images_cpu):
        # for i in range(len(gen_seq) - 1):
            # numpy 가 아니라 tensor라서 안 됨 에러도 안 남
            # gen_img1, gen_img2 = gen_images[i].clone().detach().numpy(), gen_images[i + 1].clone().detach().numpy()
            # gt_img1, gt_img2 = gt_images[i].clone().detach().numpy(), gt_images[i + 1].clone().detach().numpy()

            # tmp1 = float_to_cv8u(gen_seq[i])
            # tmp2 = float_to_cv8u(gen_seq[i + 1])

        gen_flow = optical.calc(float_to_cv8u(gen_seq), float_to_cv8u(gen_seq), None)
        gt_flow = optical.calc(float_to_cv8u(gt_seq), float_to_cv8u(gt_seq), None)

        # 채널이 두개 나오는데 광도 및 방향으로 나옴, 그 다음으로 이미지 사이즈 나옴
        gen_flow = np.transpose(gen_flow, (2, 0, 1))
        gt_flow = np.transpose(gt_flow, (2, 0, 1))

        gen_flow = normalize(gen_flow)
        gt_flow = normalize(gt_flow)

        # gen_flow = optical.calc(gen_img1, gen_img2, None)
        # gt_flow = optical.calc(gt_img1, gt_img2, None)

        gen_diff.append(gen_flow.copy())
        gt_diff.append(gt_flow.copy())

    return gen_diff, gt_diff