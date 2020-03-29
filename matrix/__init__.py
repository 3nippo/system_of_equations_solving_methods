import random
import copy
import math
from collections import namedtuple


class Matrix:
    __Shape__ = namedtuple('MatrixShape', ['rows', 'columns'])

    class DimensionError(Exception):
        def __str__(self):
            return "Matrices are not matched"

    class RowAssignmentError(Exception):
        def __init__(self, n):
            self.__length = n

        def __str__(self):
            return f"Wrong row: it should be instance of list and of length {self.__length}"

    def __init__(self, m=None, n=None, elems=None):
        n = n or m

        if not m:
            self.__A = []
            return

        if not elems:
            new_self = Matrix.unit_matrix(m, n)
            self.__dict__.update(new_self.__dict__)
            return

        l_iter = (el for el in elems)

        self.__A = [[next(l_iter) for _ in range(n)] for _ in range(m)]

    def to_list(self):
        A = self
        m, n = A.shape()

        return [A[i][j] for i in range(m) for j in range(n)]

    def shape(self):
        A = self.__A
        if len(A) == 0:
            return Matrix.__Shape__(0, 0)
        return Matrix.__Shape__(len(A), len(A[0]))

    def swap(self, swap_index_1, swap_index_2, axis=0):
        m, n = self.shape()
        A = self.__A

        if axis == 0:
            for i in range(n):
                A[swap_index_1][i], A[swap_index_2][i] = A[swap_index_2][i], A[swap_index_1][i]
        else:
            for i in range(m):
                A[i][swap_index_1], A[i][swap_index_2] = A[i][swap_index_2], A[i][swap_index_1]

    def get_index_abs_max(self, i, start=0, axis=0):
        m, n = self.shape()
        A = self.__A

        result = start

        if axis == 0:
            for j in range(start + 1, n):
                if abs(A[i][j]) > abs(A[i][result]):
                    result = j
        else:
            for j in range(start + 1, m):
                if abs(A[j][i]) > abs(A[result][i]):
                    result = j

        return result

    def get_index_max(self, i, start=0, axis=0):
        m, n = self.shape()
        A = self.__A

        result = start

        if axis == 0:
            for j in range(start + 1, n):
                if A[i][j] > A[i][result]:
                    result = j
        else:
            for j in range(start + 1, m):
                if A[j][i] > A[result][i]:
                    result = j

        return result

    def __call__(self, x):
        A = self

        m, n = A.shape()
        result = Matrix(m, n)

        if isinstance(x, (int, float)):
            arg = [x]
        else:
            arg = x.to_list()

        for i in range(m):
            for j in range(n):
                result[i][j] = A[i][j](arg)

        return result

    @staticmethod
    def unit_matrix(m, n=None):
        n = n or m

        elems = [1 if i == j else 0 for i in range(m) for j in range(n)]

        return Matrix(m, n, elems)

    def is_square(self):
        return self.shape().rows == self.shape().columns

    def assert_square(self):
        assert self.is_square(), "Matrix should be square"

    def max(self, axis=-1):
        if axis == -1:
            return self.max(0).max(1)[0][0]

        A = self
        m, n = A.shape()
        result = []

        if axis == 1:
            for i in range(n):
                l_max = A[0][i]

                for j in range(1, m):
                    l_max = max(l_max, A[j][i])

                result.append(l_max)

            return Matrix(1, n, result)
        else:
            for i in range(m):
                l_max = A[i][0]

                for j in range(1, n):
                    l_max = max(l_max, A[i][j])

                result.append(l_max)

            return Matrix(m, 1, result)

    def find(self, val):
        result = []

        A = self
        m, n = A.shape()

        for i in range(m):
            for j in range(n):
                if val == A[i][j]:
                    result.append((i, j))

        return result

    @staticmethod
    def random_matrix(m, n=None, a=-100, b=100):
        n = n or m

        elems = [random.randint(a, b) for _ in range(m * n)]

        return Matrix(m, n, elems)

    def __str__(self):
        res = ""

        for row in self.__A:
            for el in row:
                res += str(el) + ' '
            res += '\n'

        return res[:len(res) - 1]

    def __repr__(self):
        return str(self)

    def __getitem__(self, i):
        return self.__A[i]

    def __setitem__(self, i, val):
        if not isinstance(val, list) or self.shape().columns != len(val):
            raise Matrix.RowAssignmentError(self.shape().columns)
        self.__A[i] = val

    def get_column(self, i):
        A = self.__A
        m, _ = self.shape()
        return [A[j][i] for j in range(m)]

    def copy(self):
        return copy.deepcopy(self)

    def transpose(self):
        m, n = self.shape()
        A = self.__A

        elems = []

        for i in range(n):
            for j in range(m):
                elems.append(A[j][i])

        return Matrix(n, m, elems)

    def get_permutated(self, permutation):
        result = Matrix(*self.shape())

        for i in range(len(permutation)):
            result[i] = self[permutation[i]]

        return result

    def __mul__(self, b):
        m, n = self.shape()

        if isinstance(b, (int, float)):
            res = Matrix(m, n)

            for i in range(m):
                for j in range(n):
                    res[i][j] = self[i][j] * b

            return res

        b_m, b_n = b.shape()

        if n != b_m:
            raise Matrix.DimensionError

        b = b.transpose()
        res = Matrix(m, b_n)

        for i in range(b_n):
            for k in range(m):
                l_sum = 0

                for j in range(n):
                    l_sum += self[k][j] * b[i][j]

                res[k][i] = l_sum

        return res

    def __add__(self, b):
        m, n = self.shape()
        b_m, b_n = b.shape()

        if m != b_m or n != b_n:
            raise Matrix.DimensionError

        res = Matrix(m, n)

        for i in range(m):
            for j in range(n):
                res[i][j] = self[i][j] + b[i][j]

        return res

    def __sub__(self, b):
        m, n = self.shape()
        b_m, b_n = b.shape()

        if m != b_m or n != b_n:
            raise Matrix.DimensionError

        res = Matrix(m, n)

        for i in range(n):
            for j in range(n):
                res[i][j] = self[i][j] - b[i][j]

        return res

    def __neg__(self):
        return Matrix(*self.shape(), [-el for el in self.to_list()])


class Norm:
    # for symmetric matrix
    @staticmethod
    def out_of_diagonal_norm(A):
        n, _ = A.shape()

        l_sum = 0

        for i in range(n):
            for j in range(i):
                l_sum += A[i][j] * A[i][j]

        return 2 * math.sqrt(l_sum)

    @staticmethod
    def column_norm(A):
        norm = float('-inf')
        m, n = A.shape()

        for i in range(n):
            current = 0

            for j in range(m):
                current += abs(A[j][i])

            norm = max(norm, current)

        return norm

    @staticmethod
    def R_infinity_norm(V):
        norm = float('-inf')
        m, n = V.shape()

        assert n == 1, 'Wrong usage'

        for i in range(m):
            norm = max(norm, abs(V[i][0]))

        return norm


class TriDiagonalMatrix(Matrix):
    def __init__(self, m=None, elems=None):
        super().__init__(m, 3, elems)

    def to_Matrix(self):
        m, _ = self.shape()
        A = self

        elems = []

        for j in range(1, 3):
            elems.append(A[0][j])

        for j in range(2, m):
            elems.append(0)

        for i in range(1, m - 1):
            for j in range(i - 1):
                elems.append(0)

            for el in A[i]:
                elems.append(el)

            for j in range(i + 2, m):
                elems.append(0)

        for j in range(m - 2):
            elems.append(0)

        for j in range(2):
            elems.append(A[m - 1][j])

        return Matrix(m, m, elems)

    def __mul__(self, B):
        if isinstance(B, TriDiagonalMatrix):
            raise NotImplementedError

        A = self.to_Matrix()

        return A * B
