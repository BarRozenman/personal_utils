import time

import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds=0, minutes=0):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    try:
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds + minutes * 60)
        try:
            yield
        finally:
            signal.alarm(0)
    except TimeoutException as e:
        print("Timed out!")


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print(
                "[%s]" % self.name,
            )
        print("Elapsed: %s" % (time.time() - self.tstart))


def timeit_decorator(method, return_seconds=False, return_minutes=False):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        run_time_sec = te - ts
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int(run_time_sec)
        else:
            print("%r  %2.2f s" % (method.__name__, run_time_sec))
        if return_minutes:
            run_time_minutes = run_time_sec / 60
            return result, run_time_minutes
        elif return_seconds:
            return result, run_time_sec
        else:
            return result

    return timed
