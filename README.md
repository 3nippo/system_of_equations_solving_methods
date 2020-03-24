# system_of_equations_solving_methods
Here we have two modules: matrix and equation.

# Matrix
Implementation of matrices with m rows and n columns. Contains derived class TriDiagonalMatrix. It represents (!) tridiagonal matrices. Values are stored as [[a_1, b_1, c_1], ..., [a_m, b_m, c_m]], where a_1 = 0, c_m = 0.

# Equations
By now contains only analytic solution through LU decomposition.

1) LU decomposition

We can represent any nondegenerate square matrix A (mxn) as multiplication of lower triangular and upper triangular matrices LU. It takes O(m^3) operations.  But why? After we got LU decomposition we can:
- Get analytic solution of system of equations with O(k * m^2) operations, where k is number of columns in matrix of unknown values.
- Get inverse matrix A^-1 with O(m^2) operations
- Get determinant of matrix A with O(m) operations.

The error is optimized as far as possible by rearranging the rows during the LU decomposition (https://en.wikipedia.org/wiki/LU_decomposition)

2) Sweep method

Allows solve equations where A is tridiagonal matrix with O(m) operations.

3) Iteration methods

Allow solve linear equations with given error.

- Simple iterations method --- simple (!) method. No additional profits.
- Zeydel method. It is faster then simple one because simple one computes all components of X^(k + 1) based on components of X^(k) only, when Zeydel's method use new computed values of X^(k + 1) to compute the next ones if he can.
