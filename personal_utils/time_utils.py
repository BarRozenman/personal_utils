import time


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
        else:
            return result

    return timed


