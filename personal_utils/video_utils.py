import os
from pathlib import Path
from typing import Tuple, Dict, List
import subprocess as sp

import cv2
import skvideo.io

from personal_utils import file_utils
from personal_utils.file_utils import append2file_name
import copy
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, gridspec, cm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from moviepy.video.io.VideoFileClip import VideoFileClip
from send2trash import send2trash
from personal_utils.file_readers import read_video

from personal_utils.flags import flags
from personal_utils.time_utils import timeit_decorator

from personal_utils.image_utils import fig2array


@timeit_decorator
def extract_frames_from_videos_folder(
        input_video_folder_path: str,
        output_frames_folder_path: str,
        extraction_fps: int = None,
        num_of_frame_to_extract: int = None,
):
    """extract frame from whole directory tree to the same folder tree structure"""
    success = file_utils.duplicate_directory_tree(
        input_video_folder_path, output_frames_folder_path
    )
    videos_paths = file_utils.get_all_videos_paths_in_dir(input_video_folder_path)
    if not success:
        logging.getLogger(__name__).debug("failed to duplicate directory structure")
        return
    for videos_count, curr_video_path in enumerate(
            videos_paths
    ):  # todo move this loop to external function of extract frames
        try:
            relative_path_to_curr_video = (
                Path(curr_video_path).relative_to(input_video_folder_path).parent
            )
            curr_video_output_frames_path = (
                f"{output_frames_folder_path}/{relative_path_to_curr_video}"
            )
            break_video2frames(
                curr_video_path,
                curr_video_output_frames_path,
                extraction_fps,
                num_of_frame_to_extract,
            )

        except:
            logging.getLogger(__name__).error(
                f'failed reading video "{curr_video_path}"'
            )
            continue


def break_video2frames(
        curr_video_path: str,
        output_frames_folder_path: str,
        extraction_fps: int,
        num_of_frame_to_extract: str,
):
    p = Path(curr_video_path)
    curr_video_name = p.name
    vidcap = cv2.VideoCapture(curr_video_path)
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps

    if extraction_fps is None:
        extraction_fps = original_fps
    if num_of_frame_to_extract is None:
        num_of_frame_to_extract = int(duration * extraction_fps)
    else:
        num_of_frame_to_extract = int(num_of_frame_to_extract)
    frame_write_success = True
    for count, sec in enumerate(
            np.linspace(0, duration - 0.1, num_of_frame_to_extract)
    ):  # we cant extract frame fto the last second of the video (out of range)
        mili_sec = int(sec * 1000)
        frame_path = f"{output_frames_folder_path}/{Path(curr_video_name).with_suffix('')}_idx_{count}.jpg"
        if os.path.exists(frame_path):
            logging.getLogger(__name__).warning(f"{frame_path} already exists")
            continue
        vidcap.set(cv2.CAP_PROP_POS_MSEC, mili_sec)
        frame_read_success, image = vidcap.read()
        if (frame_read_success is False) or (frame_write_success is False):
            continue
        frame_write_success = cv2.imwrite(frame_path, image)  # save frame as JPG file


def extract_video_snippet(
        input_video_folder_path, output_folder_path, cut_times_file, cut_video_len=2
):
    """cut all videos in a dir if a json file with the cut times exists
    , it will cut 2 seconds for each snippet by default
    cut_times_file: is a json in this format:
    {video name: [times to start cutting],
    "1.mp4": [16],
     "2.mp4": [8, 13, 25.2, 28.6, 37, 47, 54, 62, 75, 93, 117, 141],
     "3.mp4": [138, 185, 315, 334, 340, 355, 370, 418, 433, 492, 531, 584],
     "4.mp4": [],
    """
    if os.path.exists(cut_times_file) and flags.use_cache:
        logging.getLogger(__name__).info(
            f"using cached cut_times_file - {cut_times_file}"
        )

    else:
        cut_times_file(cut_times_file)
    cut_times_dict = json.load(open(cut_times_file, "r"))
    videos_names = [
        x for x in sorted(os.listdir(input_video_folder_path)) if x.endswith(".mp4")
    ]
    for count, curr_video_name in enumerate(videos_names):
        time_list = cut_times_dict[curr_video_name]
        for count1, time in enumerate(time_list):
            input_video_path = input_video_folder_path + "/" + curr_video_name
            output_video_path = (
                f"{output_folder_path}/{curr_video_name[:-4]}_{count1}_snippet.mp4"
            )

            with VideoFileClip(input_video_path) as video:
                cut_vid = video.subclip(time, time + cut_video_len)
                cut_vid.write_videofile(output_video_path, audio_codec="aac")


def gen_cut_time_data_df(output_folder_path, cut_times_file="cut_times_file.csv"):
    """for snippet extraction it will generate the proper time to cut from original video"""
    df = pd.DataFrame(
        columns=["file_name", "original_video_name", "cut_timestamp", "meaning"]
    )
    cut_times_dict = json.load(open(cut_times_file, "r"))
    for meaning in os.listdir(output_folder_path):
        files = os.listdir(f"{output_folder_path}/{meaning}")
        for file_name in files:
            original_video_name = file_name.split("_")[0] + ".mp4"
            curr_timestamp_index = file_name.split("_")[1]
            cut_timestamp = cut_times_dict[original_video_name][
                int(curr_timestamp_index)
            ]
            df = df.append(
                {
                    "file_name": file_name,
                    "original_video_name": original_video_name,
                    "cut_timestamp": [cut_timestamp, cut_timestamp + 2],
                    "meaning": meaning,
                },
                ignore_index=True,
            )
    df.to_csv(f"{output_folder_path}/record_snippets.csv", index=False)


def resize_all_videos_in_folder_tree(dir):
    pass
    # glob.glob()
    # moviepy.video.fx.all.resize(clip, newsize=None, height=None, width=None, apply_to_mask=True)[source]


def generate_video_from_frames_array(frames_array: np.ndarray, video_path: str, fps=10):
    """
    Args:
        frames_array:  np.ndarray that contain the data of the frame themselves
        video_path:
        fps:
        delete_frames_files:
    """
    size = (frames_array.shape[1], frames_array.shape[0])
    out = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )  # may need to use *'MP4V' instead of *"mp4v" for Windows OS

    for i in range(frames_array.shape[3]):
        out.write(frames_array[:, :, :, i])
    out.release()
    print("Created video:", f"file:///{os.getcwd()}/{video_path}")


def generate_video_from_frames_paths(
        frames_list: List[str],
        video_path: Union[str, Path],
        fps=10,
        delete_frames_files=False,
):
    """combine images given as paths in frames_list to a .mp4 video
    frames_list: List[str] should contain a list of (preferably absolute) paths of existing images ordered as axpcered to apear in the video.
    video_path: str should
    delete_frames_files:bool, move frames to Trash bin
    """
    img_array = []
    size = None
    for filename in frames_list:
        img = cv2.imread(str(filename))
        if delete_frames_files is True:
            send2trash(str(filename))
        if img is None:
            logging.error(
                f"file-{filename} could not be read as an image, skipping file"
            )
            continue
        height, width, layers = img.shape
        if size is not None and size != (width, height):
            logging.error("not all images are the same size, failed to generate video")
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=size
    )  # may need to use *'MP4V' instead of *"mp4v" for Windows OS

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Created video:", f"file:///{os.getcwd()}/{video_path}")


@timeit_decorator
def generate_emotional_analysis_frames(
        frames_index_file_path: str = None,
        index_file_df: pd.DataFrame = None,
        frame_amount: int = None,
        cmap: str = "jet",
        cols2plot: Tuple = tuple([]),
        save_frames_and_get_frames_paths=False,
):
    """
    takes a file with path to image and an emotional analysis, and create a video that shows the emotional analysis
     throughout time with the video itself
    Args:
        frames_index_file_path: str, a CSV file must contain the column "img_path" with valid existing paths to images
        index_file_df: pandas.DataFrame,  a DataFrame with the same requirements as "frames_index_file_path"
        frame_amount: sets the amount of frame to use from the index_file (mostly for quick tests)
        cmap: str, choose the colormap
        cols2plot: list,which columns to include in the final emotional analyses video

    Returns:
        list of paths to frames files or numpy array contains all the video data

    """
    if frames_index_file_path:
        df = pd.read_csv(frames_index_file_path)
    else:
        df = index_file_df
    if isinstance(frame_amount, int):
        df = df.iloc[:frame_amount, :]
    if cols2plot is None:
        cols2plot = list(df.columns)
        cols2plot.remove("img_path")
    files_paths = df["img_path"]
    cols2remove = set(df.columns) - set(cols2plot)
    labels_num = len(cols2plot)
    [df.drop([i], axis=1, inplace=True) for i in cols2remove]
    df.columns = df.columns.str.title()

    def generate_frames():
        vmin = np.min(df.values[~np.isnan(df.values)])
        vmax = np.max(df.values[~np.isnan(df.values)])
        for count, img_path in enumerate(files_paths):
            fig = plt.figure()
            gs = gridspec.GridSpec(10 * labels_num, 2 * labels_num)

            frame_path = Path.cwd() / f"Frame{count}.png"
            if frame_path.exists():
                # yield frame_name
                pass  # add overwrite options later
            curr_df = copy.deepcopy(df)
            curr_df.iloc[count + 1:, :] = np.nan
            for inner_count, i in enumerate(curr_df.iteritems()):
                up_lim = inner_count * 5
                button_lim = (inner_count * 5) + 4
                ax = plt.subplot(gs[up_lim:button_lim, :])
                g = sns.heatmap(
                    i[1].values.reshape(-1, 1).T,
                    cbar=False,
                    cmap=getattr(cm, cmap),
                    xticklabels=False,
                    vmin=vmin,
                    vmax=vmax,
                    yticklabels=[i[0]],
                    ax=ax,
                )
                g.set_yticklabels([i[0]], rotation=30)
            img_ax = plt.subplot(gs[inner_count * 6 + 1:, :])
            img = plt.imread(img_path)
            img_ax.set_axis_off()
            img_ax.imshow(img)
            fig.subplots_adjust(
                bottom=0.03, top=0.93, left=0.22, right=0.85, wspace=0.1, hspace=0.2
            )
            cb_ax = fig.add_axes([0.87, 0.5, 0.03, 0.45])
            fig.tight_layout(pad=0)

            cbar = matplotlib.colorbar.ColorbarBase(
                cb_ax, ticks=[0, 0.5, 1], cmap=getattr(cm, cmap)
            )
            cbar.ax.set_yticklabels(["Low", "Medium", "High"], rotation=45)

            curr_fig_arr = fig2array(fig)
            if save_frames_and_get_frames_paths == True:
                fig.savefig(frame_path)
                plt.close("all")
                yield str(frame_path), None
            else:

                plt.close("all")
                yield None, curr_fig_arr

    frames_paths = []
    figs_arr = None
    frame_generator = generate_frames()
    for frame in frame_generator:
        frames_paths.append(frame[0])
        if figs_arr is None and frame[1] is not None:
            # plt.imshow(frame[1])
            # plt.show()
            figs_arr = np.expand_dims(frame[1], axis=3)
        elif frame[1] is not None:
            figs_arr = np.concatenate(
                (figs_arr, np.expand_dims(frame[1], axis=3)), axis=3
            )
        else:
            continue
    if save_frames_and_get_frames_paths:
        return frames_paths
    else:
        return figs_arr


def subsample_and_write(filename, out_filename, n_steps):
    """only works for short videos"""
    video_mat = skvideo.io.vread(filename)  # returns a NumPy array
    video_mat = video_mat[::n_steps]  # subsample
    skvideo.io.vwrite(out_filename, video_mat)


def split_video_and_write(filename, out_filename, start, end):
    """only works for short videos"""
    video_mat = skvideo.io.vread(filename)  # returns a NumPy array
    video_mat = video_mat[..., start:end]  # subsample
    skvideo.io.vwrite(out_filename, video_mat)


def get_every_x_frame_down_sampling(original_fps: int, target_fps: int) -> int:
    """use when you want to downsample the video to a lower fps
    Examples
    --------
    input_path = path/to/video
    video_details = get_video_details(input_path)
    frame_count = video_details['frame_count']
    original_fps = video_details['fps']
    write_every_x_frame = get_every_x_frame_down_sampling(original_fps, target_fps)

    for idx, frame in enumerate(video):
        if not idx % write_every_x_frame == 0:
            continue # skipping the frame not in the target fps
        ### do sometime with frame
        ### write frame to video

    """
    every_x_frame = int(original_fps / target_fps) if target_fps < original_fps else 1
    return every_x_frame


def set_video_cap_to_start_from_frame(video: cv2.VideoCapture, frame_idx: int):
    """ inplace video cap object  with to tart iterating from a certain frame index"""
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    return video


def get_video_details(file_name: str) -> Dict[str, int]:
    cap = cv2.VideoCapture(file_name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    details_dict = {'fps': fps, 'width': width, 'height': height, 'frame_count': frame_count, 'duration': duration}
    return details_dict


def combine_output_files(input_videos_paths: List[str], output_path: str, remove_original_files=False):
    # Create a list of output files and store the file names in a txt file
    with open("list_of_output_files.txt", "w") as f:
        for t in input_videos_paths:
            f.write("file {} \n".format(t))

    # use ffmpeg to combine the video output files
    output_path = str(Path(output_path).with_suffix('.mp4')) if not '.' in output_path else output_path
    ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + output_path
    sp.Popen(ffmpeg_cmd, shell=True).wait()

    if remove_original_files:
        for f in input_videos_paths:
            os.remove(f)
    os.remove("list_of_output_files.txt")


if __name__ == '__main__':
    pass
