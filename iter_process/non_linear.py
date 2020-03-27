from iter_process import __IterProcess__
from linear import Linear
from matrix import Norm


class NonLinear(__IterProcess__):
    def __init__(self, init_x, error):
        super().__init__(error)
        self.init_x = init_x

    def simple_iteration(self, reduced_func_vec, norm=Norm.R_infinity_norm):
        return self.__iter_process__(
            NonLinear.__GetCurrent__.simple_iteration,
            [reduced_func_vec],
            lambda x_last, x_current, l_norm: l_norm(x_last - x_current),
            [norm]
        )

    def Newton_method(self, func_vec, Jacobi_matrix, norm=Norm.R_infinity_norm):
        return self.__iter_process__(
            NonLinear.__GetCurrent__.Newton,
            [self.error, func_vec, Jacobi_matrix],
            lambda x_last, x_current, l_norm: l_norm(x_last - x_current),
            [norm]
        )

    # all funcs should take x_last first
    class __GetCurrent__:
        @staticmethod
        def Newton(last, error, func_vec, Jacobi_matrix):
            right_part = -func_vec(last)

            return Linear(error).zeydel_method(  # give right part as init value
                Jacobi_matrix(last),
                right_part
            )[0] + last

        @staticmethod
        def simple_iteration(last, reduced_func_vec):
            return reduced_func_vec(last)