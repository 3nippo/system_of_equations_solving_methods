from equation.iter_process import __IterProcess__
import math
from matrix import Matrix, Norm
import collections

class Linear(__IterProcess__):
    def __run_iter_process(self, get_next, A, B, norm, init_x=None):
        return self.__iter_process__(
            get_next,
            *self.__calc_args_and_set_init_x__(A, B, norm, init_x)  # args_curr, diff func and args_diff
        )

    def simple_iteration(self, *args, norm=Norm.column_norm, **kwargs):
        return self.__run_iter_process(
            Linear.__GetCurrent__.simple_iteration,
            *args,
            norm,
            **kwargs
        )

    def zeydel_method(self, *args, norm=Norm.column_norm, **kwargs):
        return self.__run_iter_process(
            Linear.__GetCurrent__.zeydel,
            *args,
            norm,
            **kwargs
        )

    def relax_method(self, *args, omega, norm=Norm.column_norm, **kwargs):
        return self.__run_iter_process(
            Linear.__GetCurrent__.gen_relax(omega),
            *args,
            norm,
            **kwargs
        )

    def infimum_iterations_num(self, A, B, method='simple', norm=Norm.column_norm):
        Al, Bt = Linear.__preparation__(A, B)

        Al_norm = norm(Al)

        assert Al_norm < 1, "A do not satisfy sufficient condition"

        log = math.log
        divisor = log(Al_norm)

        if method == 'zeydel':
            upper_part = Al.copy()
            n = upper_part.shape().rows

            for i in range(n):
                for j in range(i):
                    upper_part[i][j] = 0

            divisor = log(norm(upper_part))

        return math.ceil((log(self.error) - log(norm(Bt)) + log(1 - Al_norm)) / divisor - 1)

    @staticmethod
    def non_zero_diagonal(A, B):
        m, n = A.shape()
        
        perm = [0] * m

        for i in range(m):
            j = A.get_index_abs_max(i)

            perm[j] = i

        result = A.get_permutated(perm), B.get_permutated(perm)

        k = min(m, n)
        
        for i in range(k):
            assert result[0][i][i] != 0, "Row permutation failed"
        
        return result

    @staticmethod
    def __preparation__(A, B):
        m, n = A.shape()
        
        assert A.is_square(), "A should be square"

        A, B = Linear.non_zero_diagonal(A, B)
        
        Al = Matrix(m, n)

        for i in range(m):
            for j in range(n):
                if i == j:
                    Al[i][j] = 0
                else:
                    Al[i][j] = -A[i][j] / A[i][i]

        Bt = Matrix(m, 1)

        for j in range(B.shape().columns):
            for i in range(m):
                Bt[i][j] = B[i][j] / A[i][i]

        return Al, Bt

    def __calc_args_and_set_init_x__(self, A, B, norm, init_x=None):
        Al, Bt = Linear.__preparation__(A, B)
        
        self.init_x = init_x.copy() or Bt

        Al_norm = norm(Al)
        error_const = Al_norm / (1 - Al_norm) if Al_norm < 1 else 1

        def difference(x_last, x_current, norm, error_const):
            return error_const * norm(x_last - x_current)

        return [Al, Bt], difference, [norm, error_const]

    # all funcs should take x_last first
    class __GetCurrent__:
        @staticmethod
        def simple_iteration(last, A, B):
            return B + A * last

        @staticmethod
        def gen_relax(omega):
            def relax(last, A, B):
                current = last.copy()

                n, k = A.shape().columns, B.shape().columns

                for i in range(k):
                    for j in range(n):
                        current_cell = omega * B[j][i]

                        for h in range(n):
                            current_cell += omega * current[h][i] * A[j][h]

                        current[j][i] = current_cell

                return current
            
            return relax
        
        zeydel = gen_relax.__get__(object)(1)

    class BadInputError(Exception):
        def __str__(self):
            return "Input matrix of equation is degenerate"
