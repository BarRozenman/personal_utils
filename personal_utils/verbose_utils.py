import copy
from matplotlib import pyplot as plt



class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class verbose_manager():
    @property
    def vprint(self):
        from flags import flags

        if flags.verbose:
            def _vprint(*args, **kwargs):
                print(*args, **kwargs)
        else:
            _vprint = lambda *a, **k: None  # do-nothing function if flags.verbose:
        return _vprint
    @property
    def vplt(self):
        from flags import flags
        if flags.verbose:
            _vplt = plt
        else:
            _vplt = copy.deepcopy(objectview({k: lambda *a, **k: None for k in plt.__dict__.keys()}))
        return _vplt


