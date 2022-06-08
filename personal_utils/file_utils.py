import copy
import datetime as dt
import hashlib
import logging
import os
import pathlib
import platform
import re
import shutil
import stat
import sys
import warnings
from collections import OrderedDict, Counter
from glob import glob
from os.path import basename
from pathlib import Path
from shutil import copyfile
from typing import Iterable, Dict, Union, List

import numpy as np
import pandas as pd
import scipy.io
import wget
from PIL import Image
from natsort import natsorted


def update_renaming_doc(input_dir: str, sink_dir: str, df_renaming_doc: pd.DataFrame):
    """

    Args:
        input_dir: the dir path of the images before the transformation
        sink_dir:  the dir path of the images after the transformation
        df_renaming_doc: DataFrame with two columns :[original_path,new_path]

    Returns:

    records the renaming DataFrame and if an exiting renaming_doc.csv file already exits then a new column will be added to it"""
    if set(df_renaming_doc.columns) != {"original_path", "new_path"}:
        raise Exception(
            'argument "df_renaming_doc" must include the future columns: [original_path,new_path]'
        )

    if Path(input_dir).is_dir():
        input_renaming_doc_path = Path(input_dir) / "renaming_doc.csv"
    else:
        input_renaming_doc_path = copy.deepcopy(input_dir)
    if Path(sink_dir).is_dir():
        sink_renaming_doc_path = Path(sink_dir) / "renaming_doc.csv"
    else:
        sink_renaming_doc_path = copy.deepcopy(sink_dir)

    if Path(input_renaming_doc_path).exists() and Path(sink_renaming_doc_path).exists():
        raise Exception(
            "cannot use sink and input dirs renaming doc together must remove on of them"
        )
    elif Path(input_renaming_doc_path).exists():
        existing_renaming_doc_path = input_renaming_doc_path
    elif Path(sink_renaming_doc_path).exists():
        existing_renaming_doc_path = sink_renaming_doc_path
    else:
        existing_renaming_doc_path = None
    if existing_renaming_doc_path is None:
        df_renaming_doc.to_csv(sink_renaming_doc_path, index=False)
    else:
        existing_renaming_doc_df = pd.read_csv(existing_renaming_doc_path)
        """making sure we aligning the right column with as index"""
        if set(df_renaming_doc["original_path"]) == set(
            existing_renaming_doc_df.iloc[:, -1]
        ):
            new_df = existing_renaming_doc_df.set_index(
                existing_renaming_doc_df.iloc[:, -1]
            )
        elif set(df_renaming_doc["original_path"]) == set(
            existing_renaming_doc_df.iloc[:, 0]
        ):
            new_df = existing_renaming_doc_df.set_index(
                existing_renaming_doc_df.iloc[:, 0]
            )
        else:
            raise Exception(
                "cond not find matching paths in existing  renaming_doc.csv "
            )
        new_col_name = f"new_path_{len(new_df.columns)}"
        new_df[new_col_name] = None
        new_df.loc[df_renaming_doc["original_path"], new_col_name] = df_renaming_doc[
            "new_path"
        ].values
        new_df.to_csv(sink_renaming_doc_path, index=False)
        return new_df
    return df_renaming_doc


def get_latest_file(regex_path: str) -> str:
    """
    Returns:
        the latest file in the current directory using regex_path as regex
    Examples
    regex_path = "path/to/folder/*.jpg"
    latest_jpg_file_in_dir = get_latest_file(regex_path)
    """
    return str(
        max(
            list(Path(regex_path).parent.glob(Path(regex_path).name)),
            key=os.path.getctime,
        )
    )


def download_youtube_video(url: str, file_path):
    import pytube

    youtube = pytube.YouTube(url)
    video = youtube.streams.get_highest_resolution()

    output_folder = os.path.dirname(file_path)
    output_file = os.path.basename(file_path)
    video.download(output_path=output_folder, filename=output_file)


def download_files_and_write_to_individual_folders(
    links_list: List, sink_dir: str = ".", write_to_individual_folders=True
):
    """the function will get a list of links (for example links to videos on a public s3 bucket)
    download and save each of them inside individual folder on "sink_dir" directory"""

    logger = logging.getLogger(__name__)
    if not os.path.exists(sink_dir):
        try:
            os.mkdir(sink_dir)
        except Exception as e:
            logger.error(
                f'the given "sink_dir" directory does not exists, could not create it - {e} '
            )
    if not isinstance(links_list, (list, np.ndarray)):
        logger.error(
            'the given "links_list" is not type list or numpy array, please give a valid input'
        )

        return None
    for count, link in enumerate(links_list):
        p = Path(link)
        file_name = p.name
        file_name_wu_suffix = p.with_suffix("").name
        if write_to_individual_folders:
            file_path = f"{sink_dir}/{file_name_wu_suffix}/{file_name}"
            curr_dir_path = f"{sink_dir}/{file_name_wu_suffix}"
        else:
            file_path = f"{sink_dir}/{file_name}"
            curr_dir_path = f"{sink_dir}"
        if os.path.exists(file_path):
            print(f"skipped file {file_name} already exists")
            continue
        if not os.path.exists(curr_dir_path):
            Path(curr_dir_path).mkdir(exist_ok=True, parents=True)
        try:
            if "youtube" in link:
                download_youtube_video(url=link, file_path=file_path)
            else:
                wget.download(link, out=file_path)
            print("downloaded", file_name)
        except Exception as e:
            print(f"failed {file_name} with exception {e}")


def get_env_brv_dataset() -> str:
    os_dataset_dir = os.getenv("BRV_DATASETS")
    return os_dataset_dir


def normalize_dataset_path(dataset_path) -> str:
    dataset_path = str(dataset_path) if isinstance(dataset_path, Path) else dataset_path
    if not isinstance(dataset_path, str):
        return
    if not os.path.exists(dataset_path):
        env_datasets = Path(get_env_brv_dataset())
        if env_datasets is not None:
            dataset_path = env_datasets / dataset_path
            if (
                not os.path.exists(dataset_path)
                and not Path(dataset_path).parent.exists()
            ):
                dataset_path = None
        else:
            dataset_path = None
    if str(dataset_path).endswith("data"):
        dataset_path = Path(dataset_path).parent

    return str(dataset_path)


def split_filename(name):  # todo we should use pathlib instead probably
    filename, file_extension = os.path.splitext(name)
    if filename.find(".") > 0:
        filename, file_extension = os.path.splitext(filename)
    return basename(filename)


def split_full_path(name: str) -> List:
    base_folder, file_ext = os.path.split(name)
    if file_ext.find(".") > 0:
        basename, ext = os.path.splitext(file_ext)
    else:
        basename = ""
        ext = ""
    return [base_folder, basename, ext]


def get_all_images_paths_in_dir(path: Union[str, Path], valid_extensions: List = None):
    if not isinstance(path, (str, Path)):
        raise Exception("give string of an existing path as input")
    if valid_extensions is None:
        valid_extensions = (
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.PNG",
            "*.JPG",
            "*.tiff",
            "*.jfif",
            "*.webp",
        )

    elif isinstance(valid_extensions, list):
        valid_extensions = [
            f"{ext}" if ext.startswith(".") or ext.startswith("*") else f".{ext}"
            for ext in valid_extensions
        ]
        valid_extensions = [
            f"{ext}" if ext.startswith("*.") else f"*{ext}" for ext in valid_extensions
        ]
    else:
        logging.getLogger(__name__).error("please give a valid list of extensions")
        return
    all_imgs_paths = set(
        sum([glob(f"{path}/**/{x}", recursive=True) for x in valid_extensions], [])
    )  # adding here unique since JPG and jpg could be interpret as the same string
    all_imgs_paths = natsorted(
        x.replace("\\", "/")
        for x in all_imgs_paths
        if "dataignore" not in Path(x).parts and Path(x).is_file()
    )
    # this is the best solution I found for Windows to linux computability (pathlib did not help)

    return all_imgs_paths


def get_all_media_paths_in_dir(path: Union[str, Path], valid_extensions: List = None):
    all_media_paths = (
        get_all_images_paths_in_dir(path, valid_extensions)
        + get_all_videos_paths_in_dir(path, valid_extensions)
        + get_all_audio_paths_in_dir(path, valid_extensions)
    )
    return all_media_paths


def get_media_paths_from_names(dir_path, names):
    """get the absolute paths of a list of name of files"""
    all_media_paths = get_all_media_paths_in_dir(dir_path)
    d = {Path(p).name: p for p in all_media_paths}
    paths = [d[curr_name] for curr_name in names if curr_name in d]
    return paths
    pass


def get_all_audio_paths_in_dir(dir_path: str, valid_extensions: List = None):
    if valid_extensions is None:
        valid_extensions = "*.mp3", "*.wav"
    elif isinstance(valid_extensions, list):
        valid_extensions = [
            f"{ext}" if ext.startswith(".") or ext.startswith("*") else f".{ext}"
            for ext in valid_extensions
        ]
        valid_extensions = [
            f"{ext}" if ext.startswith("*.") else f"*{ext}" for ext in valid_extensions
        ]
    else:
        logging.getLogger(__name__).error("please give a valid list of extensions")
        return
    all_audio_paths = sum(
        [glob(f"{dir_path}/**/{x}", recursive=True) for x in valid_extensions], []
    )
    all_audio_paths = natsorted(
        x.replace("\\", "/")
        for x in all_audio_paths
        if "dataignore" not in Path(x).parts and Path(x).is_file()
    )
    return all_audio_paths


def get_all_videos_paths_in_dir(dir_path: str, valid_extensions: List = None):
    if valid_extensions is None:
        valid_extensions = "*.mp4", "*.mov", "*.avi"
    elif isinstance(valid_extensions, list):
        valid_extensions = [
            f"{ext}" if ext.startswith(".") or ext.startswith("*") else f".{ext}"
            for ext in valid_extensions
        ]
        valid_extensions = [
            f"{ext}" if ext.startswith("*.") else f"*{ext}" for ext in valid_extensions
        ]
    else:
        logging.getLogger(__name__).error("please give a valid list of extensions")
        return
    all_videos_paths = sum(
        [glob(f"{dir_path}/**/{x}", recursive=True) for x in valid_extensions], []
    )
    all_videos_paths = natsorted(
        x.replace("\\", "/")
        for x in all_videos_paths
        if "dataignore" not in Path(x).parts and Path(x).is_file()
    )
    return all_videos_paths


def get_git_revision(base_path):
    git_dir = pathlib.Path(base_path) / ".git"
    with (git_dir / "HEAD").open("r") as head:
        ref = head.readline().split(" ")[-1].strip()
    with (git_dir / ref).open("r") as git_hash:
        return git_hash.readline().strip()


def read_matlab_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    return mat


def transform_mat_files_to_jpg(folder_path: str):
    import matplotlib.pyplot as plt

    mat_files = glob(folder_path + "/*.mat")
    for i in mat_files:
        im_arr = read_matlab_mat_file(i)["v"]
        plt.imsave(f"{i.rsplit('.', 1)[0] + '.jpg'}", im_arr)


def delete_all_files_in_folder_and_subfolders(path):
    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            os.remove(os.path.join(root, file))


def query_yes_no(question: str, default: str = "no") -> bool:
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def transform_all_images_to_jpg_in_dir(dir_path: str):
    ans = query_yes_no(
        f'are you sure you want to rename all images in directory {dir_path} with ".jpg" suffix'
    )
    if not ans:
        return

    img_paths = get_all_images_paths_in_dir(dir_path)
    for img in img_paths:
        if Path(img).suffix == ".jpg":
            continue
        im = Image.open(img)
        rgb_im = im.convert("RGB")
        rgb_im.save(Path(img).with_suffix(".jpg"))
        os.remove(img)


def duplicate_directory_tree(input_path, output_path, exist_ok=False):
    """duplicating directory structure (folders only, without files) to a new give path, if the given output path exists and
    contains any files nothing will happen until a dir without files is given"""
    if [
        Path(dir).relative_to(input_path)
        for dir in (glob(f"{input_path}/**/", recursive=True))
    ] == [
        Path(dir).relative_to(output_path)
        for dir in (glob(f"{output_path}/**/", recursive=True))
    ] and exist_ok:
        logging.getLogger(__name__).info(
            f"directory tree already exists and match the input dir tree and exist_ok is True, skipping {duplicate_directory_tree.__name__} "
        )

        return

    def ignore_files_func(directory, files):
        ret = []
        if os.path.exists(output_path):
            ret.append(output_path)
        if os.listdir(directory) != 0:
            ret.append(directory)
        ret = ret + ([f for f in files if os.path.isfile(os.path.join(directory, f))])
        return ret

    has_files_flag = False

    for dirpath, dirnames, files in os.walk(output_path):
        if files:
            has_files_flag = True
            print(f'"{dirpath}" has files {files}')
            break

    if has_files_flag and not exist_ok:
        warnings.warn(
            f'ERROR "{output_path}" already exists and contains files please give path that do not contains files'
        )
        print(
            f'ERROR "{output_path}" already exists and contains files please give path that do not contains files'
        )

        return None
    elif Path(output_path).exists() and not has_files_flag:
        rmtree(output_path, keep_root_folder=False)

    shutil.copytree(
        input_path,
        output_path,
        ignore=lambda directory, files: [
            dir for dir in [directory] if os.listdir(dir) != 0
        ]
        + [f for f in files if os.path.isfile(os.path.join(directory, f))],
    )

    return True


def delete_empty_folders_in_dir_tree(
    root: str, keep_root_folder: bool = False, dryrun=False
):
    """recursive Function to remove empty folders in order to clear directories"""

    def recursive_empty_dir_func(dir):
        if not os.path.isdir(dir):
            return

        # remove empty sub-folders
        files = os.listdir(dir)
        if len(files):
            for f in files:
                fullpath = os.path.join(dir, f)
                if os.path.isdir(fullpath):
                    recursive_empty_dir_func(fullpath)

        # if folder empty, delete it
        files = os.listdir(dir)
        if len(files) == 0:
            if dryrun:
                print("dry run whould have removed:", dir)
            else:
                print("Removing empty folder:", dir)
                os.rmdir(dir)

    recursive_empty_dir_func(root)
    if keep_root_folder and not os.path.exists(root):
        os.mkdir(root)


def write_files_to_folders_by_label(
    labels: pd.Series, files_dir: str = None, sink_path: str = "."
):
    """copy files of each category in a new folder named after the relevant category/label name,
        can give the files paths or the files names and the directory they are in (no required if a path is given
         instead of merely the file name)

    Parameters
    ---------
    labels(pd.Series): a Series with file name or path as index and the category (str) as values
    files_dir: the path to the folder that contains the file to be separated to different folders
    sink_path: the root directory in which the folder of each category will be generated with the corresponding files

    Examples
    --------
    from clustering algorithms:

    data =pd.read_csv(data_file_path)
    paths =pd.read_csv(files_paths.csv)
    Z = hierarchy.ward(data.values)
    clustering_labels = hierarchy.cut_tree(Z, n_clusters=10).squeeze()
    clusters_series = pd.Series(clustering_labels,index=paths)
    f_utils.write_files_to_folders_by_label(clusters_series, sink_path=sink_path)

    from CSV file:

    input_dir ='/path/to/input_dir'
    output_dir = '/path/to/output_dir'
    df = pd.read_csv('/path/to/index_file.csv')
    series = df['category']
    series.index = df['file_name']
    write_files_to_folders_by_label(series,input_dir,output_dir)

    Notes
    --------
    "labels" variable should be  a pandas Series in the following format (file name as index and category name as values):
    img_(1).jpg     category_1
    img_(2).jpg     category_1
    img_(3).jpg     category_2

    "labels" variable could also contains an exiting paths of the images you want to move
    /path/to/file/img_(1).jpg     category_1
    /path/to/file/img_(2).jpg     category_1
    /path/to/file/img_(3).jpg     category_2

    you could even use a mix of paths and names if necessary...
    img_(1).jpg                   category_1
    /path/to/file/img_(2).jpg     category_1
    img_(3).jpg                   category_2

    """
    if files_dir:
        """get files path from directory using input files names"""
        files_paths = get_all_images_paths_in_dir(files_dir)
        files_names = [Path(i).name for i in files_paths]
        files_name_path_dict = OrderedDict(zip(files_names, files_paths))
    else:
        files_name_path_dict = None

    files_paths_labels = pd.Series()
    for file, label in labels.items():
        if Path(file).exists():
            file_path = file
        else:
            file_path = files_name_path_dict.get(file)
        files_paths_labels.at[file_path] = label

    for i in np.unique(labels):
        Path(f"{sink_path}").mkdir(exist_ok=True)
        p = Path(f"{sink_path}/{i}")
        p.mkdir(exist_ok=True)
        curr_label_files = files_paths_labels[files_paths_labels == i]
        [
            copyfile(file_path, p.joinpath(Path(file_path).name))
            for file_path in curr_label_files.index
        ]


def generate_hash_from_file(file_path):
    """generates a hash form the binary content of any file, can be helpful when comparing large amounts for files"""
    binary_file = open(file_path, "rb").read()
    hash_ = hashlib.md5(binary_file).hexdigest()
    return hash_


def format_stimuli_files_names(
    data_dir: str = None, data: List = None, rename: bool = False, sink_dir=None
):  # todo instead of overwriting use a sink dir to save results
    """
    reformat_stimuli_files
    actions:
    1. replace whitespace with underscore
    2. add the run index to the beginning of the file name
    3. save the new files in a sink_dir/
    4. returns the new files names

    Example
    ------
    datasets_dir = 'PATH/TO/datasets'
    subjects_datasets_dict = get_all_subjects_datasets(datasets_dir)
    for k, v in subjects_datasets_dict.items():
        format_stimuli_files_names(v)
    # it will format for example file "1/EmoImages (6).jpg" -> "1/1_EmoImages_(6).jpg" for example by so giving every file its unique name
    """
    if sink_dir is None:
        pass
        # os.mkdir('reformat_stimuli_files')
    if data_dir:
        media = get_all_media_paths_in_dir(data_dir)
    elif data:
        media = data
        rename = False
    else:
        return
    new_path_list = []
    for path in media:
        if "EmoImages" not in path:
            raise Exception("trying to changes name to a non mri scan images")
        p = Path(path)
        name = str(p.name).replace(" ", "_")
        new_path = p.parent / name
        new_path_list.append(str(new_path))
        if rename:
            os.rename(path, new_path)

    return new_path_list


def is_empty(dir):
    if not Path(dir).exists():
        logging.warning("directory doesnt exists")
        return
    if any(Path(dir).iterdir()):
        empty_flag = False
    else:
        empty_flag = True
    return empty_flag


def find_modified_files(
    dir: str = ".", minutes_ago: int = None, ignore_type: Union[List[str], str] = None
):
    if minutes_ago is None:
        minutes_ago = 5
    if ignore_type is None:
        ignore_type = [".py", ".pyc"]
    if isinstance(ignore_type, str):
        ignore_type = [ignore_type]
    if not isinstance(ignore_type, Iterable):
        logging.error('"ignore_type" must be string or Iterable')
    mod_files = []
    now = dt.datetime.now()
    ago = now - dt.timedelta(minutes=minutes_ago)
    for root, dirs, files in os.walk(dir):
        for fname in files:
            path = os.path.join(root, fname)
            if Path(path).suffix in ignore_type:
                continue
            st = os.stat(path)
            mtime = dt.datetime.fromtimestamp(st.st_mtime)
            if mtime > ago:
                mod_files.append(path)
    return mod_files


def find_new_and_modified_files(
    glob_kwargs,
    prev_dir_files: List,
    dir: str = ".",
    minutes_ago=None,
    ignore_type=None,
):
    new_files = set(glob(**glob_kwargs)) - set(prev_dir_files)
    modified_files = find_modified_files(
        dir=dir, minutes_ago=minutes_ago, ignore_type=ignore_type
    )
    new_files = list(new_files) + modified_files
    return new_files


def create_unique_file_name_dir(
    dataset_path: str = None,
    files_paths: List = None,
    rename: bool = False,
    sink_dir=None,
    generate_renaming_doc=True,
):
    """
    create unique names or file by adding the parent folder to the as a prefix
    usage:
    create_unique_file_name_dir(r'C:\###\data\training',sink_dir = r'C:\####\data\training/output')
    """
    if sink_dir is None and rename is False:
        logging.error(
            'Error: need to set a sink dir or use "rename=False" arg, existing...'
        )
        return
    if sink_dir is not None:
        duplicate_directory_tree(dataset_path, sink_dir, exist_ok=True)
    else:
        sink_dir = dataset_path
    if dataset_path:
        files_paths = get_all_media_paths_in_dir(dataset_path)
    elif files_paths is None:
        logging.error(
            'Error: need to set a sink dir or use "rename=True" arg, existing...'
        )
        return

    new_path_list = []
    original_path = []
    for path in files_paths:
        p = Path(path)
        relative_path = p.relative_to(dataset_path).parent
        if (p.name).startswith(str(relative_path)):
            name = p.name
        else:
            name = str(relative_path) + "_" + p.name
        new_path = Path(sink_dir) / relative_path / name
        if rename:
            os.rename(path, new_path)
        else:
            copyfile(p, new_path)
        new_path_list.append(str(new_path))
        original_path.append(str(path))
    df = pd.DataFrame({"original_path": original_path, "new_path": new_path_list})
    update_renaming_doc(dataset_path, sink_dir, df) if generate_renaming_doc else None
    return df


def rename_dataset_to_benchmark_format(
    dataset_path: str = None,
    files_paths: List = None,
    rename: bool = False,
    sink_dir=None,
    generate_renaming_doc=True,
):
    # TODO create a dataset_utils.py and  move some of the functions there
    """renaming media files with increasing number and the folder name, results will be a guaranteed unique names for
    dataset


    original_data:

    ├── cats
    │   ├── nonamae.jpg
    │   ├── noname.jpg
    │   ├── duplicate10.jpg
    ...
    ├── dogs
    │   ├── nonamae.jpg
    │   ├── badname.jpg
    │   ├── duplicate10.jpg
    ...

    will be renamed to:

    ├── cats
    │   ├── cats_1.jpg
    │   ├── cats_2.jpg
    │   ├── cats_3.jpg
    ...
    ├── dogs
    │   ├── dogs_1.jpg
    │   ├── dogs_2.jpg
    │   ├── dogs_3.jpg
    ...
    Examples
    --------
    rename_dataset_to_benchmark_format(r'../GoogleDriveBv/cats_vs_dogs/training')
    """
    original_path_list = []
    new_path_list = []
    if sink_dir is None and rename is False:
        logging.error(
            'Error: need to set a sink dir or use "rename=False" arg, existing...'
        )
        return
    if sink_dir is not None:
        duplicate_directory_tree(dataset_path, sink_dir, exist_ok=True)
    else:
        sink_dir = dataset_path
        ans = query_yes_no(
            f"this function will rename and overwrite all the original media files in the input dataset "
            f"directory dir {dataset_path}, are you sure you want to rename all media files?"
        )
        if not ans:
            print('aborting "rename_dataset_to_benchmark_format" function')
            return
    if dataset_path:
        files_paths = get_all_media_paths_in_dir(dataset_path)
    elif files_paths is None:
        logging.error(
            'Error: need to set a sink dir or use "rename=True" arg, existing...'
        )
        return

    for folder in Path(dataset_path).iterdir():
        if not folder.is_dir():
            continue
        folder_media_paths = get_all_media_paths_in_dir(
            str(Path(dataset_path) / folder)
        )
        count = 1
        for curr_path in folder_media_paths:
            if (
                Path(curr_path).name.startswith(str(folder))
                and Path(curr_path).with_suffix("").name[-1].isdigit()
            ):
                continue
            relative_path = Path(curr_path).relative_to(dataset_path).parent
            new_path = (
                Path(sink_dir) / relative_path / f"{folder.name}_{count}"
            ).with_suffix(Path(curr_path).suffix)
            while new_path.exists():
                count = count + 1
                new_path = (
                    Path(sink_dir) / relative_path / f"{folder.name}_{count}"
                ).with_suffix(Path(curr_path).suffix)
            else:
                count = count + 1
            if rename:
                Path(curr_path).rename(new_path)
            else:
                copyfile(curr_path, new_path)
            original_path_list.append(curr_path)
            new_path_list.append(new_path)
    df = pd.DataFrame({"original_path": original_path_list, "new_path": new_path_list})
    update_renaming_doc(dataset_path, sink_dir, df) if generate_renaming_doc else None
    rename_dataset_to_benchmark_format()
    return df


def move_media_to_parent_dir(root_path):
    """move all media inside subfolders of root directory to the root directory"""
    folders_paths = [Path(x) for x in Path(root_path).glob("*/")]
    for folder_path in folders_paths:
        files_paths = get_all_media_paths_in_dir(folder_path)
        for counter, file_path in enumerate(files_paths):
            new_path = Path(file_path).parents[0].with_name(Path(file_path).name)
            Path(file_path).replace(new_path)


def find_files_abs_paths_by_name(files_names: Union[str, List], root_dir: str) -> Dict:
    """

    Args:
        files_names: list of files names (or just one string of file name) to be searched in the root directory
        root_dir: the root in which the search will be preformed recursively

    Returns:

    """
    # if the file is present in current directory,
    # then no need to specify the absolute path
    logger = logging.getLogger(__name__)

    if isinstance(files_names, str):
        files_names = [files_names]
    files_dict = dict.fromkeys(
        files_names
    )  # must be initionlized inorder to keep the order of the dict the same as the input list

    for root, dirs, files in os.walk(Path(root_dir)):
        for name in files:
            if name in files_names:
                if files_dict[name] is not None:
                    logger.warning(
                        f"found multiple files with the same name under root directory: {files_dict[name]} and {os.path.abspath(os.path.join(root, name))}"
                    )
                files_dict[name] = os.path.abspath(os.path.join(root, name))
                if None not in files_dict.values():
                    logger.info("all files paths was found")
                    break
    if None in files_dict.values():
        msg = f"could not find all requested images paths, could not find files named {[k for k, v in files_dict.items() if v is None]}"
        logger.error(msg)
        raise Exception(msg)
    return files_dict


def is_url(s: str) -> bool:
    """[summary]

    Args:
        s (str): file path or proper http url

    Returns:
        bool: true iff s is url
    """

    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )

    return re.match(regex, s) is not None


def lower_case_all_media_files_in_dir(dir_path: str):
    """
    get  all media files in a directory recuse and remain all of them to the same name but in lower case,
    used for datasets management purposes

    Args:
        dir_path:

    Returns:

    """
    media_paths = get_all_media_paths_in_dir(dir_path)
    ans = query_yes_no(
        f'the function "lower_case_all_media_files_in_dir" is going to OVERRIDE! (rename to lower '
        f"case) the following media files: {media_paths} are you sure you want to rename all those "
        f"file to lower case?"
    )
    if not ans:
        return
    for path in media_paths:
        Path(path).replace(Path(path).with_name(Path(path).name.lower()))


def find_duplicate_media_files(dir_path: str):
    """
    find file that has identical content (for example images with different name but exactly the same otherwise)
    in a folder tree
    """

    media_files = get_all_media_paths_in_dir(dir_path)
    df = pd.DataFrame(columns=["file_path", "hash", "count"])
    """create a DataFrame fo hash and path of each file in fmri dataset and subjects dataset"""

    for file_path in media_files:
        curr_hash = generate_hash_from_file(file_path)
        df = df.append({"file_path": file_path, "hash": curr_hash}, ignore_index=True)
    count_obj = Counter(df["hash"])
    for idx, row in df.iterrows():
        row["count"] = count_obj[row["hash"]]
    df = df.drop("hash", axis=1)[df["count"] > 1]
    df.to_csv("duplicates_files_counter.csv", index=False)
    if len(df) > 0:
        print("duplicates are:")
        print(df)
    else:
        print("there are no duplicates media files in given directory")
    return df


def get_duplicate_media_files_dict(dir_path_1: str, dir_path_2: str = None) -> Dict:
    """
    find file that has identical content (for example images with different name but exactly the same otherwise)
    in a folder tree, and create a (key, value) or (media_path,identical_media_path) dict
    if only "dir_path_1" argument is given the path will be set within the given folder, if both argument are set,
    the dict will find duplicated between those two directories (/dir_path_1/img_90.jpg,/dir_path_2/img_100.jpg)

    after verifying that there are not duplicates within each given dir itself.
    """
    # media_files = get_all_media_paths_in_dir(dir_path)
    # df = pd.DataFrame(columns=["file_path", "hash", "count"])
    # """create a DataFrame fo hash and path of each file in fmri dataset and subjects dataset"""
    #
    # for file_path in media_files:
    #     curr_hash = generate_hash_from_file(file_path)
    #     df = df.append({"file_path": file_path, "hash": curr_hash}, ignore_index=True)
    # count_obj = Counter(df["hash"])
    # for idx, row in df.iterrows():
    #     row["count"] = count_obj[row["hash"]]
    # df = df.drop("hash", axis=1)[df["count"] > 1]
    # df.to_csv("duplicates_files_counter.csv", index=False)
    # print("duplicates are:")
    # print(df)
    # return df
    pass


def append2file_name(path: Union[str, Path], append: Union[str, Path]) -> str:
    append = str(append)
    path = str(path)
    append = "_" + append  if not append.startswith("_") else append
    res = Path(path).with_name(Path(path).stem + str(append) + Path(path).suffix)
    return str(res)


def prepend2file_name(path: Union[str, Path], prepend: Union[str, Path]) -> str:
    prepend = str(prepend)
    path = str(path)
    prepend = prepend + "_" if not prepend.endswith("_") else prepend
    res = Path(path).with_name(
        str(prepend)+str(Path(path).stem)+str(Path(path).suffix))
    return str(res)


def rmtree(top, keep_root_folder=False):
    """delete all file and folder in a directory"""

    if not os.path.exists(top):
        return
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            if platform.system() == "Linux":
                os.rmdir(f"{root}/{name}")
            elif platform.system() == "Windows":
                os.system('rmdir /S /Q "{}"'.format(os.path.join(root, name)))
    if not keep_root_folder:
        os.rmdir(top)


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
