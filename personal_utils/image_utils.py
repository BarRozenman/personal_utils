from typing import Union

import numpy as np
from PIL import Image
import json
import logging
import os
import pathlib
from pathlib import Path
from shutil import copyfile
from typing import List, Tuple, Union

import PIL
import cv2
import matplotlib.cm as cm
import pandas as pd
import scipy.misc
from PIL import Image

from .ProgressBar import print_progress_bar
from .exceptions import ImageResizingException
from .file_utils import get_all_images_paths_in_dir, update_renaming_doc
from .flags import flags


from . import file_utils

def new_coordinates_after_resize_img(original_size:Tuple, new_size:Tuple, original_coordinate:Tuple) -> Tuple:
  original_size = np.array(original_size)
  new_size = np.array(new_size)
  original_coordinate = np.array(original_coordinate)
  xy = original_coordinate/(original_size/new_size)
  x, y = int(xy[0]), int(xy[1])
  return x, y

def image_tracking(im: np.ndarray = None, img_path: str = None) -> np.ndarray:
    """@cvar
    this function will find the anomalies in an image by using tracking methods
    """
    if im is None and isinstance(img_path, str):
        im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    im = scipy.ndimage.gaussian_filter(im, sigma=1, mode="wrap")
    if False:
        plt.imshow(im)
        plt.show()
        plt.imshow(im1 - im)
        plt.title(np.sum(im1 - im) / 10 ** 6)
        plt.show()
    im[im < 100] = 0
    vs = 30
    first = np.zeros((*im.shape, 2))
    for i in range(vs, im.shape[0] - vs):
        for j in range(vs, im.shape[1] - vs):
            if im[i, j] > 0:
                lx = np.zeros((8, 1))
                ly = np.zeros((8, 1))
                pmx = np.mean(im[i - vs : i + 1, j])
                lx[0] = 0
                ly[0] = -pmx
                ppx = np.mean(im[i : i + vs + 1, j])
                lx[1] = 0
                ly[1] = ppx
                pmy = np.mean(im[i, j - vs : j + 1])
                lx[2] = -pmy
                ly[2] = 0
                ppy = np.mean(im[i, j : j + vs + 1])
                lx[3] = ppy
                ly[3] = 0
                tempmat = im[i - vs : i, j : j + vs]
                tempmat = np.flip(tempmat, axis=0)
                tempv = tempmat.diagonal()
                ppm45 = np.mean(tempv)
                lx[4] = -(ppm45) * 0.8
                ly[4] = (ppm45) * 0.8
                tempmat = im[i : i + vs, j : j + vs]
                tempv = tempmat.diagonal()
                ppp45 = np.mean(tempv)
                lx[5] = (ppp45) * 0.8
                ly[5] = (ppp45) * 0.8
                tempmat = im[i - vs : i :, j - vs : j]
                tempmat = np.flip(tempmat, axis=1)
                tempmat = np.flip(tempmat, axis=0)
                tempv = tempmat.diagonal()
                pmm45 = np.mean(tempv)
                lx[6] = -pmm45 * 0.8
                ly[6] = -pmm45 * 0.8
                tempmat = im[i : i + vs + 1, j - vs : j]
                tempmat = np.flip(tempmat, axis=1)
                tempv = tempmat.diagonal()
                pmp45 = np.mean(tempv)
                lx[7] = (pmp45) * 0.8
                ly[7] = -(pmp45) * 0.8
                covariance = np.cov(np.hstack((lx, ly)).T)
                [eigenval, eigenvec] = np.linalg.eig(covariance)
                eigenval = np.flip(eigenval, axis=0)
                eigenval = np.real_if_close(np.diag(eigenval))  # [::-1]
                values = eigenval[eigenval != 0]
                E1 = np.argmax(values == max(values))
                eigenvec = -np.flip(eigenvec, axis=1)
                first[i, j, :] = (eigenvec[:, E1]) * ((im[i, j]) / 255)
            else:
                first[i, j, :] = [0, 0]
        if isinstance(img_path, str):
            print_progress_bar(
                i, len(range(vs, im.shape[0] - 1)), f"analyzing {img_path}"
            )
        else:
            print_progress_bar(i, len(range(vs, im.shape[0] - 1)), "analyzing image")

    u = first[:, :, 0]
    v = first[:, :, 1]
    cmap = None
    vmin, vmax = None, None
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    rgb_im = sm.to_rgba(v, bytes=True)
    gray_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGBA2GRAY)
    return gray_im


def fourier_power(image: np.ndarray = None) -> np.ndarray:
    """
    Apply fourier transform and centering on given image.
    If image contains multiple channels - apply fourier on each of them seperately
    @param img:
    @return:
    """
    if image is None:
        return image

    channels = cv2.split(image)

    full_channels_power_img_array = np.zeros((image.shape))
    for counter, image in enumerate(channels):

        img_fft = np.fft.fft2(image) / 255
        power = np.log1p(np.abs(img_fft))
        power = np.fft.fftshift(power)
        if np.nan in power or -np.inf in power or np.inf in power:
            logging.getLogger(__name__).warning(
                "the image is probably grayscale please convert to to rgb"
            )
            zero_loc = np.where(
                np.logical_or(np.isnan(power), np.isinf(power), np.isneginf(power))
            )
            power[zero_loc[0], zero_loc[1]] = 0
            power[zero_loc[0], zero_loc[1]] = np.nanmean(
                power
            )  # make it regulae number and dont stand out in the analysis

        if len(channels) == 1:
            full_channels_power_img_array = power
        else:
            full_channels_power_img_array[:, :, counter] = power

    return full_channels_power_img_array


def fourier_phase(image: np.ndarray = None) -> np.ndarray:
    """
    Apply fourier transform and centering on given image.
    If image contains multiple channels - apply fourier on each of them seperately
    @param img:
    @return:
    """
    if image is None:
        return image

    channels = cv2.split(image)

    full_channels_phase_img_array = np.zeros((image.shape))
    for counter, image in enumerate(channels):

        img_fft = np.fft.fft2(image) / 255
        phase_img = np.angle(img_fft)
        phase_img = np.fft.fftshift(phase_img)
        if np.nan in phase_img or -np.inf in phase_img or np.inf in phase_img:
            # logging.getLogger(__name__).warning('the image is probably grayscale please convert to to rgb')
            zero_loc = np.where(
                np.logical_or(
                    np.isnan(phase_img), np.isinf(phase_img), np.isneginf(phase_img)
                )
            )
            # phase_img[zero_loc[0], zero_loc[1]] = 0
            phase_img[zero_loc[0], zero_loc[1]] = np.nanmean(
                phase_img
            )  # make it regulae number and dont stand out in the analysis

        if len(full_channels_phase_img_array.shape) == 2:
            full_channels_phase_img_array = phase_img
        else:
            full_channels_phase_img_array[:, :, counter] = phase_img

    plot_command = flags.debug
    if plot_command:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(full_channels_phase_img_array[:, :, 0])
        plt.show()
    return full_channels_phase_img_array


def write_imgs_to_csv(all_imgs_paths: List, score_vec=None, sink_dir: str = "."):
    """a function that uses cpbd method to estimate image sharpness"""

    Path(sink_dir).mkdir(exist_ok=True, parents=True)
    output_file_name = f"{sink_dir}/frames_data.csv"
    col_names = ["img_path", "sharpness_score"]
    if score_vec is None:
        score_vec = [None] * len(all_imgs_paths)
    df = pd.DataFrame(dict(zip(col_names, [all_imgs_paths, score_vec])))
    df.to_csv(output_file_name, index=False)
    return df


def resize_and_pad_all_imgs_in_folder_tree(
    input_folder_path: str,
    sink_dir: str,
    new_size: Tuple = (1440, 810),
    instructions_file_path: str = None,
    pad_img: bool = False,
) -> None:
    """

    Args:
        input_folder_path:
        sink_dir:
        new_size:
        instructions_file_path:
        pad_img:

    Returns:

    This function in used to normalize images for an MRI scan. transform (by [resizing and cropping] or [resizing and
    padding]) all images in folder tree structure, while maintaining the folder structure. by default the upper part
    of the images is kept in the final image, this can be controlled by using a "instructions file" json (see Notes),
    in which the images you wish to cropped the bottom appear (if image doesnt appear in the "instructions file" we
    use the the upper part to crop as default



    Example
    -------
    resize_all_imgs_in_folder_tree(
    "/path/to/images/folder",
    sink_dir=""/path/to/output/folder"",
    pad_img=True)

    Notes
    ------
    if you want to use a instructions_file.json it should be in this format:

    /instructions_file.json:

    {
    "<folder_name>/<image_name>":"bottom",
    "<folder_name>/<image_name>":"bottom"
    }

    **the final image will be save as ".jpg" file

    """
    new_path_list = []
    original_path = []
    file_utils.duplicate_directory_tree(input_folder_path, sink_dir)
    all_images_paths = file_utils.get_all_images_paths_in_dir(input_folder_path)
    if instructions_file_path is None:
        instrct = dict()
    else:
        instrct = json.load(open(instructions_file_path, "r"))

    for count, img_path in enumerate(all_images_paths):
        crop_flag = None
        img_folder = pathlib.PurePath(img_path).parts[-2]
        if img_folder + "/" + os.path.basename(img_path) in instrct.keys():
            crop_flag = instrct[
                pathlib.PurePath(img_path).parts[-2] + "/" + os.path.basename(img_path)
            ]
        img = Image.open(img_path)
        if pad_img:
            img = add_padding_to_image(img, new_size)
        else:
            img = resize_and_crop_img_to_fit_new_size(img, new_size, crop_flag)

        path = pathlib.PurePath(img_path).with_suffix(".jpg")
        new_path = Path(sink_dir) / path.relative_to(input_folder_path)
        img.save(new_path)
        original_path.append(str(img_path))
        new_path_list.append(str(new_path))
        if flags.debug:
            import matplotlib.pyplot as plt

            plt.imshow(img)
            plt.show()
            print("plot image")
    df = pd.DataFrame({"original_path": original_path, "new_path": new_path_list})

    update_renaming_doc(input_folder_path, sink_dir, df)


def resize_and_crop_img_to_fit_new_size(
    img: PIL.Image, target_size: Tuple, crop_flag: str = None
) -> PIL.Image:
    """

    this function will try to fit the input "img" as best as possible to the "target_size" by resize and cropping when
    necessary
    (the upper part of an image will by cropped by default unless input "crop_flag" is different not None)

    Args:
        img (PIL.Image): input image
        target_size (tuple): the new file of the image
        crop_flag (str): 'top' or 'bottom'

    Returns: (PIL.Image)

    """

    target_width, target_height = target_size
    width, height = img.size
    target_proportions = target_height / target_width
    img_proportions = height / width

    if img_proportions >= target_proportions:
        wpercent = target_width / float(width)
        hsize = int((float(height) * float(wpercent)))
        img = img.resize((target_width, hsize), PIL.Image.ANTIALIAS)
    else:
        hpercent = target_height / float(height)
        wsize = int((float(width) * float(hpercent)))
        img = img.resize((wsize, target_height), PIL.Image.ANTIALIAS)

    temp_img_shape = img.size
    _, height_diff = [
        temp_img_shape[0] - target_width,
        temp_img_shape[1] - target_height,
    ]

    if crop_flag is None:
        left, top, right, bottom = 0, 0, target_width, target_height
    elif crop_flag == "bottom":
        left, top, right, bottom = (
            0,
            height_diff,
            target_width,
            height_diff + target_height,
        )
    else:
        left, top, right, bottom = 4 * [None]
    img = img.crop((left, top, right, bottom))  # ((left, top, right, bottom))

    return img


def add_padding_to_image(
    img: PIL.Image, target_size: Tuple, background_color: Tuple = (0, 0, 0)
) -> PIL.Image:
    """
    pad image with background color so it will fit a certain target shape.

    it will add background to both sides (upo and down of left and right, depending on the images size and target
    size) until the images will be at the target size.

    if the images is to large it will be resize to a smaller size before padding

    Args:
        img:
        target_size:
        background_color:

    Returns:
        padded_img
    """
    width, height = img.size
    target_width, target_height = target_size
    target_proportions = target_height / target_width
    img_proportions = height / width
    if img_proportions < target_proportions:
        wpercent = target_width / float(width)
        hsize = int((float(height) * float(wpercent)))
        img = img.resize((target_width, hsize), PIL.Image.ANTIALIAS)
    else:
        hpercent = target_height / float(height)
        wsize = int((float(width) * float(hpercent)))
        img = img.resize((wsize, target_height), PIL.Image.ANTIALIAS)

    temp_img_shape = img.size
    width_diff, height_diff = [
        target_width - temp_img_shape[0],
        target_height - temp_img_shape[1],
    ]

    if img_proportions < target_proportions:
        padded_img = Image.new(img.mode, target_size, background_color)
        padded_img.paste(img, (0, height_diff // 2))
        return padded_img
    else:
        padded_img = Image.new(img.mode, target_size, background_color)
        padded_img.paste(img, (width_diff // 2, 0))
        return padded_img


def gray_scale_entire_dir(dir):
    res = input(
        f"this will override all images in dir {dir}! what to continue? (yes/no)"
    )
    if res != "yes":
        print("cancle operation, exiting...")
        return
    img_p = get_all_images_paths_in_dir(dir)
    im_obj = [cv2.imread(i) for i in img_p]
    gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in im_obj if i is not None]
    [cv2.imwrite(p, i) for i, p in zip(gray, img_p)]


def rename_imgs_by_to_ratio_proximity(
    target_img_size=(810, 1440), input_dir=None, output_dir=None
):
    """renaming images to index 1 2 3 ... (and generate a index file with the original name of the images)
    ordered by how similar they are to the target ratio.
    mostly used  to choose best image for MRI scan by sorting  the possible images by the proximity to the ratio of
     the MRI images ratio"""
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    target_ratio = target_img_size[0] / target_img_size[1]
    imgs_paths = file_utils.get_all_images_paths_in_dir(input_dir)
    df = pd.DataFrame(columns=["img_path", "proximity"])
    for img_path in imgs_paths:
        im = cv2.imread(img_path)
        if im.size < 300 * 300:
            continue
        curr_ratio = im.shape[0] / im.shape[1]
        df = df.append(
            {"img_path": img_path, "proximity": abs(curr_ratio - target_ratio)},
            ignore_index=True,
        )
    df = df.sort_values(by="proximity")
    df = df.reset_index(drop=True)
    for row in df.itertuples():
        idx = row.Index
        img_path = row.img_path
        im = cv2.imread(img_path)
        if im.size < 300 * 300:
            continue
        curr_ratio = im.shape[0] / im.shape[1]
        p = Path(img_path)
        ext = p.suffix
        out_p = Path(Path(output_dir), str(idx) + ext)
        copyfile(p, out_p)
    df["img_name"] = df["img_path"].apply(lambda x: Path(x).name)
    df.to_csv(Path(output_dir) / "imgs_names.csv", index_label="img_num")


def resize_image(
    image: np.ndarray, new_size: Union[Tuple, List], interpolation: int = cv2.INTER_AREA
):
    """
    same as resize_image but we can enter the new size as ndarray.shape tuple or list and it set the width and
    height (since the resize_image  argument  is transposed by default...)

    Args:
        image: numpy array with shape of (height,width,RGB)
        new_size: expediting (height,width) format, can be extracted by ndarray.shape
        interpolation: a cv2 interpolation option

    Returns:

    Examples
    --------
    resized_image  = resize_image(np.asarray(PIL.Image.open('/path/to/image_1.jpg')),(100,100))

    """
    if np.asarray(new_size).shape[0] < 2:
        raise ImageResizingException(
            '"new_size" should be a list of tuple with (height,width) values only'
        )
    elif np.asarray(image).ndim < 2:
        raise ImageResizingException(
            '"image" should be a numpy array of an image with shape (height,width,RGB)'
        )

    width = int(new_size[1])
    height = int(new_size[0])
    try:
        resized_img = cv2.resize(
            np.asarray(image), (width, height), interpolation=interpolation
        )
    except Exception as e:
        raise ImageResizingException(e)

    return resized_img


def fig2array(fig: 'plt.Figure', return_RGB=False) -> np.ndarray:
    """
    by default returns a BGR channels to conform cv2 format
    fig = plt.figure()
    image = fig2array(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    arr = np.asarray(image)[:, :, :-1]
    if not return_RGB:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)  # returning BGR
    return arr


if __name__ == "__main__":
    pass
    resize_and_pad_all_imgs_in_folder_tree(
        "/home/barroz/BrainVivo_data/datasets/friends_seinfeld_original_images_mri_struct",
        "/home/barroz/BrainVivo_data/datasets/friends_seinfeld_presented_images_mri_struct",
        pad_img=True,
    )
    # image_tracking(img_path=r"C:\Git\dev\datasets\AM_test2\original_images\e.jpg")
    # rename_imgs_by_to_ratio_proximity(input_dir=r"C:\Users\Bar Rozenman\Desktop\DANA",output_dir=r"C:\Users\Bar Rozenman\Desktop\D1")


def crop_object(
    image: Union[PIL.Image.Image, np.ndarray], box: np.ndarray
) -> Union[PIL.Image.Image, np.ndarray]:
    """
      Crops an object in an image using [left,upper,right,lower] bounding box

    Inputs:
      image: PIL image
      box: one box from Detectron2 pred_boxes
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    x_top_left = box[0]
    y_top_left = box[1]
    x_bottom_right = box[2]
    y_bottom_right = box[3]
    x_center = (x_top_left + x_bottom_right) / 2
    y_center = (y_top_left + y_bottom_right) / 2

    crop_img = image.crop(
        (int(x_top_left), int(y_top_left), int(x_bottom_right), int(y_bottom_right))
    )
    if isinstance(image, np.ndarray):
        crop_img = np.asarray(crop_img)
    return crop_img
