from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms import transforms

from personal_utils.flags import flags


def get_mean_var_rgb(img: Tensor = None, img_path: str = None):
    if img_path:
        pil_img = Image.open(img_path).convert("RGB")

    else:
        pil_img = transforms.ToPILImage()(img).convert("RGB")
    img_arr = np.array(pil_img)
    output = 6 * [None]
    for channel_idx in range(img_arr.shape[2]):
        curr_arr = img_arr[..., channel_idx]
        output[2 * channel_idx] = np.sum(curr_arr) / curr_arr.size
        output[2 * channel_idx + 1] = np.std(curr_arr) / (curr_arr.size / 1000000)
    return output


def batch_get_contrast(images: Tensor, device: str = "cuda") -> Tensor:
    lab_imgs = rgb_to_lab(images, device)[:, 0, :, :]
    max_img = F.max_pool2d(lab_imgs, 3, stride=1, padding=1)
    min_img = -F.max_pool2d(-lab_imgs, 3, stride=1, padding=1)
    contrast = (max_img - min_img) / (max_img + min_img)
    contrast[(contrast < -100) + (contrast > 100)] = torch.mean(
        contrast[(contrast > -100) * (contrast < 100)]
    )
    # get average across whole image
    average_contrast = torch.mean(contrast, dim=(1, 2))
    return average_contrast


def lab_to_rgb(lab: Tensor, device: str = "cuda") -> Tensor:
    lab_pixels = torch.reshape(lab, [-1, 3])
    # convert to fxfyfz
    lab_to_fxfyfz = (
        torch.tensor(
            [
                #   fx      fy        fz
                [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
                [1 / 500.0, 0.0, 0.0],  # a
                [0.0, 0.0, -1 / 200.0],  # b
            ]
        )
        .type(torch.FloatTensor)
        .to(device)
    )
    fxfyfz_pixels = torch.mm(
        lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device),
        lab_to_fxfyfz,
    )

    # convert to xyz
    epsilon = 6.0 / 29.0
    linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).to(device)
    exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).to(device)

    xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4 / 29.0)) * linear_mask + (
        (fxfyfz_pixels + 0.000001) ** 3
    ) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = torch.mul(
        xyz_pixels,
        torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).to(device),
    )

    xyz_to_rgb = (
        torch.tensor(
            [
                #     r           g          b
                [3.2404542, -0.9692660, 0.0556434],  # x
                [-1.5371385, 1.8760108, -0.2040259],  # y
                [-0.4985314, 0.0415560, 1.0572252],  # z
            ]
        )
        .type(torch.FloatTensor)
        .to(device)
    )

    rgb_pixels = torch.mm(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    # clip
    rgb_pixels[rgb_pixels > 1] = 1
    rgb_pixels[rgb_pixels < 0] = 0

    linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).to(device)
    exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).to(device)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
        ((rgb_pixels + 0.000001) ** (1 / 2.4) * 1.055) - 0.055
    ) * exponential_mask

    return torch.reshape(srgb_pixels, lab.shape)


def rgb_to_lab(srgb: Tensor, device="cuda") -> Tensor:
    srgb_pixels = torch.reshape(srgb, [-1, 3])

    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).to(device)
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).to(device)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
        ((srgb_pixels + 0.055) / 1.055) ** 2.4
    ) * exponential_mask

    rgb_to_xyz = (
        torch.tensor(
            [
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334],  # R
                [0.357580, 0.715160, 0.119193],  # G
                [0.180423, 0.072169, 0.950227],  # B
            ]
        )
        .type(torch.FloatTensor)
        .to(device)
    )

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(
        xyz_pixels,
        torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754])
        .type(torch.FloatTensor)
        .to(device),
    )

    epsilon = 6.0 / 29.0

    linear_mask = (
        (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).to(device)
    )

    exponential_mask = (
        (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).to(device)
    )

    fxfyfz_pixels = (
        xyz_normalized_pixels / (3 * epsilon**2) + 4.0 / 29.0
    ) * linear_mask + (
        (xyz_normalized_pixels + 0.000001) ** (1.0 / 3.0)
    ) * exponential_mask
    # convert to lab
    fxfyfz_to_lab = (
        torch.tensor(
            [
                #  l       a       b
                [0.0, 500.0, 0.0],  # fx
                [116.0, -500.0, 200.0],  # fy
                [0.0, 0.0, -200.0],  # fz
            ]
        )
        .type(torch.FloatTensor)
        .to(device)
    )
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor(
        [-16.0, 0.0, 0.0]
    ).type(torch.FloatTensor).to(device)
    # return tf.reshape(lab_pixels, tf.shape(srgb))
    return torch.reshape(lab_pixels, srgb.shape)


def get_gram_mat(style_layers: Tensor) -> List[Tensor]:
    gram_mat_list = []
    for i in style_layers:
        channels, width, height = i.shape
        i = i.squeeze().view(channels, width * height)
        gram_mat = i.matmul(i.T)
        gram_mat = gram_mat / (channels * width * height)
        gram_mat_list.append(gram_mat)
    if flags.debug:
        plt.imshow(transforms.ToPILImage()(gram_mat_list[-3]))
        plt.show()
    return gram_mat_list


def compute_total_variation_loss(img: Tensor, weight: float = 1.0):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


def unnormalize(img: Tensor, mean: float, std: float):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img
