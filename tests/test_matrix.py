import context
from matrix import Matrix

m1 = [ 4,  2,  3,  2,
       0,  8,  7,  6,
      13, 15, 11, 15]

m2 = [8, 1, 1,
      5, 3, 4,
      7, 2, 7,
      3, 4, 8]

res = [ 69,  24,  49,
       107,  62, 129,
       301, 140, 270]

A = Matrix(3, 4, elems=m1)
B = Matrix(4, 3, elems=m2)
Res = Matrix(3, elems=res)

print(A)
print()
print(A[0])
print()
print(A[0][0])
print()
print(B)
print()
print(A * B)
print()
print(Res)
print()
print(A + B.transpose())
print()
print(Matrix(0))
print()
print(Matrix(0).shape())
print()
Res[0] = [1, 1]  # should throw an exception as wrong row is assigned
