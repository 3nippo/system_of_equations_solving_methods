import context
from matrix import Matrix
from equation.iter_process.eigen_things import EigenThings
from equation.decomp import QRDecomp
import matplotlib.pyplot as plt
import numpy as np


A = Matrix(3, elems = [-1,  4, -4,
                        2, -5,  0,
                       -8, -2,  0]
)

B = Matrix(3, elems = [ 1,  3,  1,
                        1,  1,  4,
                        4,  3,  1]
)

error = 2**(-20)


def test(A, error, title):
    (values, _), _ = EigenThings(A, error).qr_algorithm()

    print(title)
    print()

    print('Error:', error)
    print()

    print("Computed eigenvalues:")
    for v in values:
        print(v)
    print()

    print("Actual eigenvalues:")
    print(
        np.linalg.eig(
            np.array(
                A.to_list()
            ).reshape((3, 3))
        )[0]
    )
    print()
    print()


test(A, error, "Variant 10")
test(B, error, "Example from manual")

errors = [2**i for i in range(5, -15, -1)]
x = []
y = []

for error in errors:
    y.append(error)
    x.append(EigenThings(A, error).qr_algorithm().iterations)

plt.plot(x, y)
plt.xlabel('iterations')
plt.ylabel('error')
plt.title('Error -- Number of iterations\n dependence')

plt.show()
