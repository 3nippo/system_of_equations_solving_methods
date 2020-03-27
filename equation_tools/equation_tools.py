from matrix import Matrix, Norm
import math


class EigenThings:
    def __sub_init__(self, A, error):
        self.__A = A.copy()
        self.__error = error

    def __init__(self, A, error):
        self.__sub_init__(A, error)

    def get_things(self):
        if '__values' not in self.__dict__:
            self.__calculate_things__()
        return self.__values, self.__vectors

    def recalculate(self, A, error):
        self.__sub_init__(A, error)
        return self.get_things()

    def __calculate_things__(self, norm=Norm.out_of_diagonal_norm):
        A = self.__A.copy()
        error = self.__error
        m, n = A.shape()

        basis = Matrix(m, n)

        atan = math.atan
        cos = math.cos
        sin = math.sin
        pi = math.pi

        iterations = 0

        while norm(A) > error:
            i, j = A.find(A.max())[0]

            phi = pi / 4 if A[i][i] == A[j][j] else 1 / 2 * atan(2 * A[i][j] / A[i][i] - A[j][j])

            U = Matrix(m, n)

            U[i][i] = cos(phi)
            U[i][j] = -sin(phi)
            U[j][i] = sin(phi)
            U[j][j] = cos(phi)

            basis = basis * U
            A = U.transpose() * A * U

            iterations += 1

        self.__values  = [A[i][i] for i in range(min(m, n))]
        self.__vectors = [Matrix(m, 1, basis.get_column(i)) for i in range(n)]
        self.__iterations = iterations

    def get_iterations(self):
        if '__iterations' not in self.__dict__:
            self.__calculate_things__()
        return self.__iterations
