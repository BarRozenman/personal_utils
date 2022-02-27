from pathlib import Path
from typing import Union


def append2file_name(path: Union[str,Path], append: Union[str,Path]) -> str:
    append = str(append)
    path = str(path)
    if not append.startswith('_'):
        append = '_' + append
    res = Path(path).with_name(Path(path).stem + str(append) + Path(path).suffix)
    return str(res)
