from dataclasses import dataclass
from datetime import datetime

# from brv_constants.flags import flags

# TODO change this
from pathlib import Path


@dataclass
class Flags(object):
    """
    class that holds global variables that can be access int any place in the run
    eval_subset_size: [bool,int], take a subset of the total dataset to run, use int to choose the subset size
    """
    debug: bool = False
    verbose: bool = False
    timestamp: str = None
    timestamp_seconds: str = None
    use_multi_process: bool = True
    overwrite: bool = False
    use_cache: bool = False
    eval_subset_size: int = None
    counter: int = 0
    mlflow_run_id: str = None
    random_seed: int = None
    clear_cache: bool = False

    def __post_init__(self):
        self.timestamp = self.get_timestamp_min_str()
        self.timestamp_seconds = self.get_timestamp_sec_str()

    def get_dict(self):
        return self.__dict__

    @staticmethod
    def use_cache_file(file_path: str) -> bool:
        """if use_cache_file is True and file exists return True """
        if Path(file_path).exists() and flags.use_cache:
            return True
        else:
            return False

    @staticmethod
    def get_timestamp_min_str():
        return datetime.now().strftime("%Y-%m-%d_%H%M")

    @staticmethod
    def get_timestamp_sec_str():
        return datetime.now().strftime("%Y-%m-%d_%H%M-%S")

    def set_flags_values(self, flags=None):
        """set the values in the ConfigReader to the flags object (if they exist in the flags object"""
        # if flags is None:
        #     flags = self
        # [
        #     setattr(flags, k, v)
        #     for k, v in self.__dict__.items()
        #     if k in flags.get_dict() and v is not None
        # ]
        # return flags
        pass


# TODO - eval_subset_size should be renamed to value of relative size (e.g. percentage)
flags = Flags()
