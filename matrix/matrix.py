import random
import copy


class MatrixDimensionError(Exception):
    def __str__(self):
        return "Matrices are not matched"


class RowAssignmentError(Exception):
    def __init__(self, n):
        self.__length = n

    def __str__(self):
        return f"Wrong row: it should be instance of list and of length {self.__length}"


class Matrix:
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

    def shape(self):
        A = self.__A
        if len(A) == 0:
            return 0, 0
        return len(A), len(A[0])

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

    @staticmethod
    def unit_matrix(m, n=None):
        n = n or m

        elems = [1 if i == j else 0 for i in range(m) for j in range(n)]

        return Matrix(m, n, elems)

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
        if not isinstance(val, list) or self.shape()[1] != len(val):
            raise RowAssignmentError(self.shape()[1])
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

    def __mul__(self, b):
        m, n = self.shape()
        b_m, b_n = b.shape()

        if n != b_m:
            raise MatrixDimensionError

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
            raise MatrixDimensionError

        res = Matrix(m, n)

        for i in range(m):
            for j in range(n):
                res[i][j] = self[i][j] + b[i][j]

        return res

    def __sub__(self, b):
        m, n = self.shape()
        b_m, b_n = b.shape()

        if m != b_m or n != b_n:
            raise MatrixDimensionError

        res = Matrix(m, n)

        for i in range(n):
            for j in range(n):
                res[i][j] = self[i][j] - b[i][j]

        return res


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
