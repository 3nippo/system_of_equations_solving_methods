from equation.iter_process import linear
from matrix import Matrix

class Laplace:
    def __init__(
        self,
        h1,
        h2,
        l1,
        l2,
        A_coefs, # ij i+1j i-1j ij+1 ij-1
        phi1,
        phi2,
        psi1,
        psi2,
        error
    ):
        N1 = int(l1 / h1) + 1
        N2 = int(l2 / h2) + 1

        A = []

        n = N1 * N2

        for i in range(1, N1 - 1):
            for j in range(1, N2 - 1):
                row = [0] * n

                row[i * N2 + j] = A_coefs[0]
                row[(i+1) * N2 + j] = A_coefs[1]
                row[(i-1) * N2 + j] = A_coefs[2]
                row[i * N2 + j+1] = A_coefs[3]
                row[i * N2 + j-1] = A_coefs[4]

                A.extend(row)
        
        B = [0] * n

        for i in range(N1):
            xi = i * h1

            row = [0] * n
            row[i * N2 + 0] = 1
            B[i * N2 + 0] = psi1(xi)
            A.extend(row)

            row = [0] * n
            row[i * N2 + N2 - 1] = 1
            B[i * N2 + N2 - 1] = psi2(xi)
            A.extend(row)

        for j in range(1, N2 - 1):
            yj = j * h2

            row = [0] * n
            row[0 * N2 + j] = 1
            B[0 * N2 + j] = phi1(yj)
            A.extend(row)

            row = [0] * n
            row[(N1-1) * N2 + j] = 1
            B[(N1-1) * N2 + j] = phi2(yj)
            A.extend(row)
        
        self.A = Matrix(n, elems=A)
        self.B = Matrix(n, 1, elems=B)

        init_x = [0] * n

        for i in range(N1):
            xi = i * h1

            for j in range(N2):
                yj = j * h2

                init_x[i * N2 + j] = (phi2(yj) - phi1(yj)) / (N1 - 1) * xi + phi1(yj)

        self.init_x = Matrix(n, 1, init_x)

        self.solver = linear.Linear(error)

    def simple(self):
        return self.solver.simple_iteration(
            self.A, 
            self.B, 
            init_x=self.init_x
        )

    def zeydel(self):
        return self.solver.zeydel_method(
            self.A, 
            self.B, 
            init_x=self.init_x
        )

    def relax(self, omega):
        return self.solver.relax_method(
            self.A, 
            self.B,
            omega=omega,
            init_x=self.init_x
        )
