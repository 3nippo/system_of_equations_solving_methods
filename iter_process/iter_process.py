from matrix import Matrix, Norm
from collections import namedtuple


class __IterProcess__:
    __Result__ = namedtuple('IterProcessResult', ['answer', 'iterations'])

    def __init__(self, error):
        self.error  = error
        self.init_x = None  # will be provided

    def __iter_process__(self, current, args_curr, difference, args_diff):
        if not self.init_x:
            raise NotImplementedError

        x_last = self.init_x
        x_current = current(x_last, *args_curr)

        iterations = 1

        while difference(x_last, x_current, *args_diff) > self.error:
            x_last    = x_current
            x_current = current(x_last, *args_curr)

            iterations += 1

        return __IterProcess__.__Result__(x_current, iterations)