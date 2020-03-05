# system_of_equations_solving_methods
Here we have two modules: matrix and equation.

# Matrix
Simple implementation of matrices with m rows and n columns.

# Equations
By now contains only analytic solution through LU decomposition.

1) LU decomposition

We can represent any nondegenerate square matrix A (mxn) as multiplication of lower triangular and upper triangular matrices LU. It takes O(m^3) operations.  But why? After we got LU decomposition we can:
- Get analytic solution of system of equations with O(k * m^2) operations, where k is number of columns in matrix of unknown values. The error is optimized as far as possible by rearranging the rows during the solution (https://en.wikipedia.org/wiki/LU_decomposition)
- Get inverse matrix A^-1 with O(m^2) operations
- Get determinant of matrix A with O(m) operations.
