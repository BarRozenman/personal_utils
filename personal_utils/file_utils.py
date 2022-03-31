import os
import platform
import stat
from pathlib import Path
from typing import Union


def append2file_name(path: Union[str, Path], append: Union[str, Path]) -> str:
    append = str(append)
    path = str(path)
    if not append.startswith('_'):
        append = '_' + append
    res = Path(path).with_name(Path(path).stem + str(append) + Path(path).suffix)
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