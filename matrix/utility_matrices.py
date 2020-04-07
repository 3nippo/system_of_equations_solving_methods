import math
from matrix import Matrix


class Householder:
    @staticmethod
    def __construct_matrix__(V):
        m, n = V.shape()

        assert n == 1, "V should be vector"

        VT = V.transpose()

        scalar_product = (VT * V).to_val()
        matrix = V * VT

        return Matrix.unit_matrix(m) - 2 / scalar_product * matrix

    @staticmethod
    def gauss_method(A, i, shift=0):
        """
        Returns Householder matrix.

        args:
            A --- matrix
            i --- number of column you want to set
                  to zero with gauss method
        """
        joint = i + shift

        vec_elems = [0 for _ in range(joint)]

        norm = 0
        m = A.shape().rows

        for j in range(joint, m):
            norm += A[j][i] * A[j][i]

        norm = math.sqrt(norm)

        vec_elems.append(
            A[joint][i] + norm * (-1 if A[joint][i] < 0 else 1)
        )

        for j in range(joint + 1, m):
            vec_elems.append(A[j][i])

        return Householder.__construct_matrix__(
            Matrix(m, 1, vec_elems)
        )

    @staticmethod
    def to_hessenberg(A):
        """
        Returns matrix A in Hessenberg form so that it is similar
        to initial A.
        """
        transformations = Matrix.unit_matrix(A.shape().rows)

        for i in range(min(*A.shape()) - 2):
            transformations = Householder.gauss_method(A, i, 1) * \
                              transformations

        return transformations.transpose() * A * transformations
