from pathlib import Path


def append2file_name(path: str, append: str) -> str:
    append = str(append)
    if not append.startswith('_'):
        append = '_' + append
    res = Path(path).with_name(Path(path).stem + str(append) + Path(path).suffix)
    return str(res)
