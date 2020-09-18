import cv2
import numpy as np


def to_grayscale(images):
    binary_images = []

    for seq in images:
        for img in seq:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            binary_images.append(img.copy())

    return binary_images


def gpu_to_cpu(images):
    cpu_images = []

    for seq in images:
        seq_images = []

        for img in seq:
            seq_images.append(np.transpose(img.clone().detach().to('cpu').numpy(), (1, 2, 0)))

        cpu_images.append(seq_images.copy())

    return cpu_images


def float_to_cv8u(array):
    min = np.min(array)
    array = array - min
    max = np.max(array)
    div = max / float(255)
    array = np.uint8(np.round(array / div))

    return array


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
        for i in range(len(gen_seq) - 1):
            # numpy 가 아니라 tensor라서 안 됨 에러도 안 남
            # gen_img1, gen_img2 = gen_images[i].clone().detach().numpy(), gen_images[i + 1].clone().detach().numpy()
            # gt_img1, gt_img2 = gt_images[i].clone().detach().numpy(), gt_images[i + 1].clone().detach().numpy()

            tmp1 = float_to_cv8u(gen_seq[i])
            tmp2 = float_to_cv8u(gen_seq[i + 1])

            gen_flow = optical.calc(float_to_cv8u(gen_seq[i]), float_to_cv8u(gen_seq[i + 1]), None)
            gt_flow = optical.calc(float_to_cv8u(gt_seq[i]), float_to_cv8u(gt_seq[i + 1]), None)

            # gen_flow = optical.calc(gen_img1, gen_img2, None)
            # gt_flow = optical.calc(gt_img1, gt_img2, None)

            gen_diff.append(gen_flow.copy())
            gt_diff.append(gt_flow.copy())

    return gen_diff, gt_diff
