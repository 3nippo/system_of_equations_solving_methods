from equation import Equation
from equation.iter_process.linear import Linear
from matrix import TriDiagonalMatrix, Matrix
from collections import namedtuple

MiddleCoefs = namedtuple('MiddleCoefs', ['A', 'B_functor'])

class Hyperbole:
    def __init__(
        self,
        a_squared,
        phi_start,
        phi_end,
        psi_1,
        psi_2,
        psi_1_sec_der,
        x_step,
        t_step,
        l,
        T
    ):
        self.a = a_squared
        self.phi_start = phi_start
        self.phi_end = phi_end
        self.psi_1 = psi_1
        self.psi_2 = psi_2
        self.psi_1_sec_der = psi_1_sec_der
        self.x_step = x_step
        self.t_step = t_step
        self.l = l
        self.T = T
        
        self.n_x_steps = int(self.l / self.x_step) + 1
        
        self.n_t_steps = int(self.T / self.t_step) + 1

        self.sigma = self.a * self.t_step**2 / self.x_step**2
    
    def __get_first_two_layaers(self):
        t_step_squared = self.t_step**2

        Ut = [[self.psi_1(j * self.x_step)  for j in range(self.n_x_steps)]]

        first_layer = []

        for j in range(self.n_x_steps):
            x = j * self.x_step
            value = self.psi_1(x) + self.psi_2(x) * self.t_step + self.a * self.psi_1_sec_der(x) * t_step_squared / 2
            first_layer.append(value)

        Ut.append(first_layer)

        return Ut

    def explicit(self):
        assert self.t_step**2 * self.a / self.x_step**2 < 1, "Squared x_step failed stability condition for a"
        
        Ut = self.__get_first_two_layaers()

        for k in range(2, self.n_t_steps):
            u_cur = [0] * self.n_x_steps

            for j in range(1, self.n_x_steps - 1):
                u_cur[j] = self.sigma * Ut[-1][j+1] + (-2 * self.sigma + 2) * Ut[-1][j] + self.sigma * Ut[-1][j-1] - Ut[-2][j]
            
            t = k * self.t_step

            u_cur[0] = self.phi_start(t)
            u_cur[-1] = self.phi_end(t)

            Ut.append(u_cur)

        return Ut

    def __implicit(self, middle_coefs, A_borders, B_borders):
        Ut = self.__get_first_two_layaers()       

        for k in range(2, self.n_t_steps):
            t = k * self.t_step

            sqr_x_step = self.x_step**2

            A_coefs = A_borders[0] 

            u_prev = Ut[-1]

            B_coefs = [B_borders[0](u_prev, t)]
            
            A_coefs += middle_coefs.A * (self.n_x_steps - 2)
            
            B_coefs += [middle_coefs.B_functor(u_prev, Ut[-2], j) for j in range(1, self.n_x_steps - 1)]

            A_coefs.extend(A_borders[1])
            
            B_coefs.append(B_borders[1](u_prev, t))

            A = TriDiagonalMatrix(self.n_x_steps, A_coefs)
            B = Matrix(self.n_x_steps, 1, elems=B_coefs)

            u_curr = Equation(A, B).sweep_method().to_list()

            Ut.append(u_curr)

        return Ut

    def implicit(self):
        A_borders = [[
            1, 
            0
        ]]
        A_borders.append([
            0,
            1
        ])
        
        def lower(u, t):
            return self.phi_start(t)

        def upper(u, t):
            return self.phi_end(t)

        B_borders = [
            lower,
            upper
        ]

        def B_functor(u_prev, u_prev_prev, j):
            return -2 * u_prev[j] + u_prev_prev[j]
        
        middle_coefs = MiddleCoefs(
            [
                self.sigma,
                (-2 * self.sigma - 1),
                self.sigma
            ],
            B_functor
        )

        return self.__implicit(middle_coefs, A_borders, B_borders)

class ParabolaDim2:
    def __init__(
        self,
        a,
        b,
        h1,
        h2,
        dt,
        l1,
        l2,
        T,
        phi1,
        phi2,
        psi1,
        psi2,
        ksi,
        f
    ):
        self.a = a
        self.b = b
        self.dt = dt
        self.h1 = h1
        self.h2 = h2
        self.l1 = l1
        self.l2 = l2
        self.T = T
        self.phi1 = phi1
        self.phi2 = phi2
        self.psi1 = psi1
        self.psi2 = psi2
        self.ksi = ksi
        self.f = f

        self.N1 = int(self.l1 / self.h1) + 1
        self.N2 = int(self.l2 / self.h2) + 1
        self.NT = int(self.T / self.dt) + 1
    
    def __calc_next(
        self,
        u_prev,
        k,
        stage_k,
        A_coefs, 
        calc_B,
        A_coefs_border, 
        calc_B_border, 
        order,
        exceptions
    ):
        if order == 1:
            ordered = [self.N2, self.N1]
            h = self.h2
            u_cur = [[] for _ in range(self.N1)]
        elif order == 2:
            ordered = [self.N1, self.N2]
            h = self.h1
            u_cur = []
        else:
            raise NotImplementedError
        
        shift = int(bool(exceptions))

        for i in range(shift, ordered[0] - shift):
            A = list(A_coefs_border[0])
            B = [calc_B_border[0](h * i, self.dt * stage_k)]

            for j in range(1, ordered[1] - 1):
                A.extend(A_coefs)
                B.append(calc_B(u_prev, i, j, k))

            A.extend(A_coefs_border[1])
            B.append(calc_B_border[1](h * i, self.dt * stage_k))

            A = TriDiagonalMatrix(ordered[1], A)
            B = Matrix(ordered[1], 1, B)

            solution = Equation(A, B).sweep_method().to_list()
            
            if order == 1:
                for i, el in enumerate(solution):
                    u_cur[i].append(el)
            else:
                u_cur.append(solution)

        if exceptions:
            if order == 1:
                for i, el in enumerate(u_cur):
                    begin = exceptions[0](el, i, stage_k)
                    end = exceptions[1](el, i, stage_k)
                    el.insert(0, begin)
                    el.append(end)
            else:
                begin = []
                end = []

                for j in range(len(u_cur[0])):
                    begin.append(exceptions[0](u_cur[0], j, stage_k))
                    end.append(exceptions[1](u_cur[-1], j, stage_k))

                u_cur.insert(0, begin)
                u_cur.append(end)

        return u_cur

    def __common_solution(
        self,
        two_A_coefs, 
        two_calc_B,
        two_A_coefs_border,
        two_calc_B_border,
        two_exceptions
    ):
        Ut = [[[self.ksi(self.h1 * i, self.h2 * j) for j in range(self.N2)] for i in range(self.N1)]]
        
        for k in range(self.NT):
            middle = self.__calc_next(
                Ut[-1],
                k,
                k + 1/2,
                two_A_coefs[0],
                two_calc_B[0],
                two_A_coefs_border[0],
                two_calc_B_border[0],
                1,
                two_exceptions[0]
            )
            
            u_cur = self.__calc_next(
                middle,
                k,
                k + 1,
                two_A_coefs[1],
                two_calc_B[1],
                two_A_coefs_border[1],
                two_calc_B_border[1],
                2,
                two_exceptions[1]
            )

            Ut.append(u_cur)

        return Ut
    
    def fractional_steps(self):
        two_A_coefs = [
            [
                self.a / self.h1**2,
                -1 / self.dt - 2 * self.a / self.h1**2,
                self.a / self.h1**2
            ],
            [
                self.b / self.h2**2,
                -1 / self.dt - 2 * self.b / self.h2**2,
                self.b / self.h2**2
            ]
        ]

        def calc_B_1(u_prev, j, i, k):
            return -self.f(i * self.h1, j * self.h2, k * self.dt)/2 \
                   \
                   -1/self.dt * u_prev[i][j]
        
        def calc_B_2(u_prev, i, j, k):
            return -self.f(i * self.h1, j * self.h2, (k+1) * self.dt)/2 \
                   \
                   -1/self.dt * u_prev[i][j]

        two_calc_B = [
            calc_B_1,
            calc_B_2
        ]
        
        two_A_coefs_border = [
            [
                [1, 0],
                [-1/self.h1, 1/self.h1]
            ],
            [
                [1, 0],
                [-1/self.h2, 1/self.h2]
            ]
        ]

        two_calc_B_border = [
            [
                self.psi1,
                self.psi2
            ],
            [
                self.phi1,
                self.phi2
            ]
        ]

        return self.__common_solution(
            two_A_coefs,
            two_calc_B,
            two_A_coefs_border,
            two_calc_B_border,
            [[], []]
        )

    def variable_direction(self):
        two_A_coefs = [
            [
                self.a / self.h1**2,
                -2 / self.dt - 2 * self.a / self.h1**2,
                self.a / self.h1**2
            ],
            [
                self.b / self.h2**2,
                -2 / self.dt - 2 * self.b / self.h2**2,
                self.b / self.h2**2
            ]
        ]

        def calc_B_1(u_prev, j, i, k):
            return -self.b/self.h2**2 * (u_prev[i][j+1] + u_prev[i][j-1]) \
                   \
                   + (2 * self.b / self.h2**2 - 2/self.dt) * u_prev[i][j] \
                   \
                   - self.f(i * self.h1, j * self.h2, (k + 1/2) * self.dt)
        
        def calc_B_2(u_prev, i, j, k):
            return -self.a/self.h1**2 * (u_prev[i+1][j] + u_prev[i-1][j]) \
                   \
                   + (2 * self.a / self.h1**2 - 2/self.dt) * u_prev[i][j] \
                   \
                   - self.f(i * self.h1, j * self.h1, (k + 1/2) * self.dt)

        two_calc_B = [
            calc_B_1,
            calc_B_2
        ]
        
        two_A_coefs_border = [
            [
                [1, 0],
                [-1/self.h1, 1/self.h1]
            ],
            [
                [1, 0],
                [-1/self.h2, 1/self.h2]
            ]
        ]

        two_calc_B_border = [
            [
                self.psi1,
                self.psi2
            ],
            [
                self.phi1,
                self.phi2
            ]
        ]

        two_exceptions = [
            [
                lambda el, i, k: self.phi1(i*self.h1, k*self.dt),
                lambda el, i, k: self.h2*self.phi2(i*self.h1, k*self.dt) + el[-1]
            ],
            [
                lambda el, j, k: self.psi1(j*self.h2, k*self.dt),
                lambda el, j, k: self.h1*self.psi2(j*self.h2, k*self.dt) + el[j]
            ]
        ]

        return self.__common_solution(
            two_A_coefs,
            two_calc_B,
            two_A_coefs_border,
            two_calc_B_border,
            two_exceptions
        )

class ParabolaDim1Kind3rd:
    def __init__(
        self,
        a,
        b,
        c,
        phi_start,
        phi_end,
        psi,
        x_step,
        t_step,
        l,
        T
    ):
        self.a = a
        self.b = b
        self.c = c
        self.phi_start = phi_start
        self.phi_end = phi_end
        self.psi = psi
        self.x_step = x_step
        self.t_step = t_step
        self.l = l
        self.T = T
        
        self.n_x_steps = int(self.l / self.x_step) + 1
        
        self.n_t_steps = int(self.T / self.t_step) + 1

        def simple_B_functor(u, j):
            return -u[j] / self.t_step 

        self.simple_implicit_coefs = MiddleCoefs(
            [
                self.a / self.x_step**2 - self.b / (2 * self.x_step),
                -1 / self.t_step - 2 * self.a / self.x_step**2 + self.c,
                self.a / self.x_step**2 + self.b / (2 * self.x_step)
            ],
            simple_B_functor
        )

        self.theta = 1/2
            
        def KrankNik_functor(u, j):
            return self.t_step * (self.theta - 1) / 2 / self.x_step**2 * (2*self.a + self.b*self.x_step) * u[j+1] + \
                   \
                   + (self.theta - 1) * (-2 * self.a * self.t_step / self.x_step**2 - 1/(self.theta - 1) + self.c * self.t_step) * u[j] + \
                   \
                   + self.t_step * (self.theta - 1) / 2 / self.x_step**2 * (2*self.a - self.b*self.x_step) * u[j-1]

        self.kranknik_implicit_coefs = MiddleCoefs(
            [
                self.t_step * self.theta / 2 / self.x_step**2 * (2*self.a - self.b*self.x_step),
                self.theta * (-2 * self.a * self.t_step / self.x_step**2 - 1/self.theta + self.c * self.t_step),
                self.t_step * self.theta / 2 / self.x_step**2 * (2*self.a + self.b*self.x_step)
            ],
            KrankNik_functor 
        )
    
    def __explicit(self, border_funcs):
        assert self.x_step**2 >= 2 * self.t_step * self.a, "Squared x_step failed stability condition for a"
        assert self.x_step >= self.t_step * self.b / 2, "Squared x_step failed stability condition for b"
        
        Ut = [[self.psi(j * self.x_step)  for j in range(self.n_x_steps)]]

        for k in range(1, self.n_t_steps):
            u_cur = [0] * self.n_x_steps
            u_prev = Ut[-1]

            sqr_x_step = self.x_step**2

            for j in range(1, self.n_x_steps - 1):
                u_cur[j] = self.t_step * (2 * self.a + self.b * self.x_step) / (2 * sqr_x_step) * u_prev[j + 1] \
                           \
                           + (self.t_step * (self.c * sqr_x_step - 2 * self.a) + sqr_x_step) / sqr_x_step * u_prev[j] \
                           \
                           + self.t_step * (2 * self.a - self.b * self.x_step) / (2 * sqr_x_step) * u_prev[j - 1]
            
            t = k * self.t_step

            u_cur[0] = border_funcs[0](u_cur, t, u_prev)
            u_cur[-1] = border_funcs[1](u_cur, t, u_prev)

            Ut.append(u_cur)

        return Ut

    def explicit_order1st(self):
        def lower(u, t, _):
            return (self.x_step * self.phi_start(t) - u[1]) / (self.x_step - 1)

        def upper(u, t, _):
            return (self.x_step * self.phi_end(t) + u[-2]) / (self.x_step + 1)

        return self.__explicit(border_funcs=[lower, upper])

    def explicit_order2nd_points3(self):
        def lower(u, t, _):
            return (2 * self.x_step * self.phi_start(t) - 4 * u[1] + u[2]) / (2 * self.x_step - 3)

        def upper(u, t, _):
            return (2 * self.x_step * self.phi_end(t) + 4 * u[-2] - u[-3]) / (2 * self.x_step + 3)

        return self.__explicit(border_funcs=[lower, upper])

    def explicit_order2nd_points2(self):
        znam_lower = self.b * self.x_step - 2 * self.a
        ksi_lower = 1 + 2 * self.a / (self.x_step * znam_lower) + self.x_step * (1 - self.c * self.t_step) / (self.t_step * znam_lower)

        def lower(u, t, u_prev):
            return (self.phi_start(t) + 2 * self.a / (self.x_step * znam_lower) * u[1] + self.x_step / (self.t_step * znam_lower) * u_prev[0]) / ksi_lower
        
        znam_upper = self.b * self.x_step + 2 * self.a
        ksi_upper = 1 + 2 * self.a / (self.x_step * znam_upper) + self.x_step * (1 - self.c * self.t_step) / (self.t_step * znam_upper)

        def upper(u, t, u_prev):
            return (self.phi_end(t) + 2 * self.a / (self.x_step * znam_upper) * u[-2] + self.x_step / (self.t_step * znam_upper) * u_prev[-1]) / ksi_upper

        return self.__explicit(border_funcs=[lower, upper])
    
    def __implicit(self, middle_coefs, A_borders, B_borders):
        Ut = [[self.psi(j * self.x_step)  for j in range(self.n_x_steps)]]

        for k in range(1, self.n_t_steps):
            t = k * self.t_step

            sqr_x_step = self.x_step**2

            A_coefs = A_borders[0] 

            u_prev = Ut[-1]

            B_coefs = [B_borders[0](u_prev, t)]
            
            A_coefs += middle_coefs.A * (self.n_x_steps - 2)
            
            B_coefs += [middle_coefs.B_functor(u_prev, j) for j in range(1, self.n_x_steps - 1)]

            A_coefs.extend(A_borders[1])
            
            B_coefs.append(B_borders[1](u_prev, t))

            A = TriDiagonalMatrix(self.n_x_steps, A_coefs)
            B = Matrix(self.n_x_steps, 1, elems=B_coefs)

            u_curr = Equation(A, B).sweep_method().to_list()

            Ut.append(u_curr)

        return Ut

    def __implicit_order1st(self, middle_coefs):
        A_borders = [[
            1 - 1 / self.x_step, 
            1 / self.x_step
        ]]
        A_borders.append([
            -1 / self.x_step, 
            1 + 1 / self.x_step
        ])
        
        def lower(u, t):
            return self.phi_start(t)

        def upper(u, t):
            return self.phi_end(t)

        B_borders = [
            lower,
            upper
        ]

        return self.__implicit(middle_coefs, A_borders, B_borders)

    def __implicit_order2nd_points3(self, middle_coefs):
        sqr_x_step = self.x_step**2
        ksi_start = 1 / (2 * self.x_step * (self.a / sqr_x_step + self.b / (2 * self.x_step))) 

        A_borders = [[
            1 - 3 / (2 * self.x_step) + ksi_start * (self.a / sqr_x_step - self.b / (2 * self.x_step)),
            (-1 / self.t_step - 2 * self.a / sqr_x_step + self.c) * ksi_start + 2 / self.x_step
        ]]

        def lower(u, t):
            return self.phi_start(t) - u[1] / self.t_step * ksi_start

        B_borders = [lower]

        ksi_end = -1 / (2 * self.x_step * (self.a / sqr_x_step - self.b / (2 * self.x_step))) 

        A_borders.append([
            -2 / self.x_step + ksi_end * (-1 / self.t_step - 2 * self.a / sqr_x_step + self.c),
            1 + 3 / (2 * self.x_step) + ksi_end * (self.a / sqr_x_step + self.b / (2 * self.x_step))
        ])
        
        def upper(u, t):
            return self.phi_end(t) - u[-2] / self.t_step * ksi_end

        B_borders.append(upper)
        
        return self.__implicit(middle_coefs, A_borders, B_borders)

    def __implicit_order2nd_points2(self, middle_coefs):
        A_borders = [[
            2 * self.a**2 / self.x_step + self.x_step / self.t_step - self.c * self.x_step - (2 * self.a**2 - self.b * self.x_step),
            -2 * self.a**2 / self.x_step
        ]]

        def lower(u, t):
            return self.x_step / self.t_step * u[0] - self.phi_start(t) * (2 * self.a**2 - self.b * self.x_step)

        B_borders = [lower]

        A_borders.append([
            -2 * self.a**2 / self.x_step,
            2 * self.a**2 / self.x_step + self.x_step / self.t_step - self.c * self.x_step + (2 * self.a**2 + self.b * self.x_step)
        ])

        def upper(u, t):
            return self.x_step / self.t_step * u[-1] + self.phi_end(t) * (2 * self.a**2 + self.b * self.x_step)

        B_borders.append(upper)

        return self.__implicit(middle_coefs, A_borders, B_borders)

    def simple_implicit_order1st(self):
        return self.__implicit_order1st(self.simple_implicit_coefs)

    def simple_implicit_order2nd_points3(self):
        return self.__implicit_order2nd_points3(self.simple_implicit_coefs)

    def simple_implicit_order2nd_points2(self):
        return self.__implicit_order2nd_points2(self.simple_implicit_coefs)

    def kranknik_implicit_order1st(self):
        return self.__implicit_order1st(self.kranknik_implicit_coefs)

    def kranknik_implicit_order2nd_points3(self):
        return self.__implicit_order2nd_points3(self.kranknik_implicit_coefs)

    def kranknik_implicit_order2nd_points2(self):
        return self.__implicit_order2nd_points2(self.kranknik_implicit_coefs)    


