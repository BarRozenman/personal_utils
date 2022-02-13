from datetime import datetime
# TODO change this

class Flags(object):
    """
    eval_subset_size: [bool,int], take a subset of the total dataset to run, use int to choose the subset size
    """

    def __init__(self, items_dict):
        for key, val in items_dict.items():
            setattr(self, key, val)

    def get_dict(self):
        return self.__dict__

    @staticmethod
    def get_timestamp_min_str():
        return datetime.now().strftime("%Y-%m-%d_%H%M")

    @staticmethod
    def get_timestamp_sec_str():
        return datetime.now().strftime("%Y-%m-%d_%H%M-%S")


TIMESTAMP = Flags.get_timestamp_min_str()
TIMESTAMP_SECONDS = Flags.get_timestamp_sec_str()
# TODO - eval_subset_size should be renamed to value of relative size (e.g. percentage)
flags = Flags(
    {
        "debug": False,
        "verbose": False,
        "timestamp": TIMESTAMP,
        "timestamp_seconds": TIMESTAMP_SECONDS,
        "use_multi_process": True,
        "overwrite": False,
        "use_cache": False,
        "eval_subset_size": None,
        "counter": 0,
        "mlflow_run_id": None,
        "random_seed": None,
    }
)