import pandas as pd
import matrix
import equation
import math
from equation.iter_process.non_linear import NonLinear


class Cauchy:
    def __init__(self, Points, a, b, step, f_true):
        self.Points = Points
        self.a = a
        self.b = b
        self.step = step
        self.f_true = f_true

    def __init_method__(self, names):
        row = {}

        row[names[0]] = self.a

        pos = [self.a]
        row['x'] = pos[0]
        row['true_y'] = self.f_true(self.a)

        for i, val in enumerate(self.Points):
            row[names[i]] = val
            pos.append(val)

        row['error'] = abs(row['y'] - row['true_y'])

        return pos, row

    def __update_method__(self, row, pos, less_pos, names):
        next_row = {'x': pos[0]}

        for i in range(len(names)):
            row[f"delta_{names[i]}"] = pos[i+1] - row[names[i]]
            next_row[names[i]] = pos[i+1]

        next_row['true_y'] = self.f_true(pos[0])
        next_row['error'] = abs(next_row['true_y'] - next_row['y'])

        runge_romberg_error = (less_pos[1] - pos[1]) / (2 ** self.__p - 1)
        next_row['runge_romberg_error'] = runge_romberg_error

        return row, next_row

    def __solve__(self,
                  update_pos,
                  equations,
                  *upd_args,
                  names=['y', 'z'],
                  custom_init=None):
        answer = []

        if custom_init:
            answer = custom_init()
            row = answer[-1]
            answer.pop()

            pos = [self.a]
            for name in names:
                pos.append(row[name])
        else:
            pos, row = self.__init_method__(names)

        while abs(pos[0] - self.b) >= 0.0000001:
            pos = update_pos(
                equations,
                pos,
                self.step,
                *upd_args
            )

            less_pos = update_pos(
                equations,
                pos,
                self.step / 2,
                *upd_args
            )

            row, next_row = self.__update_method__(row, pos, less_pos, names)

            answer.append(row)
            row = next_row

        row['true_y'] = self.f_true(self.b)
        row['error'] = abs(row['true_y'] - row['y'])

        answer.append(row)

        return pd.DataFrame(answer)

    def Euler(self, *args, **kwargs):
        self.__p = 1
        return self.__solve__(
            Cauchy.__Update_pos__.Euler,
            *args,
            **kwargs
        )

    def Runge_Kutta(self, *args, **kwargs):
        self.__p = 4
        return self.__solve__(
            Cauchy.__Update_pos__.Runge_Kutta,
            *args,
            **kwargs
        )

    def Adams(self, *args, **kwargs):
        self.__p = 4

        save_b = self.b
        self.b = self.a + 3 * self.step

        df = self.Runge_Kutta(*args, **kwargs)

        names = kwargs['names']

        help_values = []
        for name in names:
            help_values.append(df[name].values.tolist())

        equations = args[0]

        help_values = list(zip(df['x'].values.tolist(), *help_values))
        help_values = [[eq(hv) for hv in help_values] for eq in equations]

        self.a = self.b
        self.b = save_b

        def custom_init():
            return df.to_dict('records')

        kwargs['custom_init'] = custom_init

        return self.__solve__(
            Cauchy.__Update_pos__.Adams,
            *args,
            help_values,
            **kwargs
        )

    class __Update_pos__:
        @staticmethod
        def Euler(equations, pos_, step):
            pos = pos_.copy()

            for i, eq in enumerate(equations):
                pos[i+1] = pos[i+1] + step*eq(pos)

            pos[0] += step

            return pos

        @staticmethod
        def Runge_Kutta(equations, pos, step):
            coefs = [[] for _ in range(len(equations))]

            for k, eq in zip(coefs, equations):
                k.append(step*eq(pos))

            for i in range(2):
                curr_pos = pos.copy()

                curr_pos[0] += step / 2

                for i in range(len(coefs)):
                    curr_pos[i+1] += coefs[i][-1] / 2

                for k, eq in zip(coefs, equations):
                    k.append(step*eq(curr_pos))

            curr_pos = pos.copy()

            curr_pos[0] += step

            for i in range(len(coefs)):
                curr_pos[i+1] += coefs[i][-1]

            for k, eq in zip(coefs, equations):
                k.append(step*eq(curr_pos))

            deltas = [step]

            for k in coefs:
                delta = (k[0] + 2*k[1] + 2*k[2] + k[3]) / 6
                deltas.append(delta)

            return [p + d for p, d in zip(pos, deltas)]

        @staticmethod
        def Adams(equations, pos_, step, help_values):
            pos = pos_.copy()

            pos[0] += step
            approx_point = pos.copy()

            for eq, hv, i in zip(equations, help_values, range(1, len(pos))):
                mult = 55*hv[-1] - 59*hv[-2] + 37*hv[-3] - 9*hv[-4]
                approx = pos_[i] + step / 24 * mult

                approx_point[i] = approx

            for eq, hv, i in zip(equations, help_values, range(1, len(pos))):
                del hv[0]
                hv.append(eq(approx_point))

                mult = 9*hv[-1] + 19*hv[-2] - 5*hv[-3] + hv[-4]
                pos[i] = pos_[i] + step / 24 * mult

            return pos


class BoundaryVals:
    def __init__(self, a, b, step, f_true):
        self.a = a
        self.b = b
        self.step = step
        self.f_true = f_true

    def shooting_method(self,
                        left_vals,
                        missing_num,
                        difference,
                        eps,
                        equations,
                        names=['y', 'z']):
        h_curr = 0.8
        conds = left_vals.copy()

        conds[missing_num] = h_curr

        obj = Cauchy(
            conds,
            self.a,
            self.b,
            self.step,
            self.f_true
        )

        vals = obj.Runge_Kutta(equations, names=names).iloc[-1]

        last_error = difference(vals)

        h_last = h_curr
        h_curr = 1

        while True:
            conds[missing_num] = h_curr

            obj = Cauchy(
                conds,
                self.a,
                self.b,
                self.step,
                self.f_true
            )

            vals = obj.Runge_Kutta(equations, names=names).iloc[-1]

            curr_error = difference(vals)

            if curr_error < eps:
                break

            h_curr, h_last = h_curr - (h_curr - h_last) / (curr_error - last_error) * curr_error, h_curr

        return obj.Runge_Kutta(equations, names=names)

    def finite_difference(self,
                          y_a,
                          y_b,
                          der_a,
                          der_b,
                          p,
                          q,
                          f,
                          eps=0.00001,
                          a_coefs=None,
                          b_coefs=None):
        h = self.step
        N = (self.b - self.a) / h
        N = math.floor(N)

        A_elems = []
        B_elems = []

        pos = self.a

        if der_a:
            if a_coefs:
                A_elems.extend(a_coefs)
            else:
                A_elems.extend([-1/h, 1/h])

            B_elems.append(y_a)

        pos += h

        if der_a:
            A_elems.append(1 - p(pos)*h/2)

        A_elems.extend([-2+h*h*q(pos), 1+p(pos)*h/2])

        B_elems.append(h*h*f(pos))

        if not der_a:
            B_elems[-1] -= y_a*(1 - p(pos)*h/2)

        pos += h

        for k in range(2, N-1):
            A_elems.append(1 - p(pos)*h/2)
            A_elems.append(-2 + h*h*q(pos))
            A_elems.append(1 + p(pos)*h/2)

            B_elems.append(h*h*f(pos))

            pos += h

        A_elems.extend([1-p(pos)*h/2, -2+h*h*q(pos)])

        if der_b:
            A_elems.append(1+p(pos)*h/2)
            B_elems.append(h*h*f(pos))

            if b_coefs:
                A_elems.extend(b_coefs)
            else:
                A_elems.extend([-1/h, 1/h])
            B_elems.append(y_b)
        else:
            B_elems.append(h*h*f(pos) - y_b*(1+p(pos)*h/2))

        n = N-4+1+2+der_a+der_b

        applicable = abs(A_elems[0]) > abs(A_elems[1])

        for i in range(2, len(A_elems) - 2, 3):
            inner = abs(A_elems[i+1]) > abs(A_elems[i]) + abs(A_elems[i+2])
            applicable = applicable and inner

        applicable = applicable and abs(A_elems[-1]) > abs(A_elems[-2])

        triA = matrix.TriDiagonalMatrix(n, A_elems)
        A = triA.to_Matrix()
        B = matrix.Matrix(n, 1, B_elems)

        self.A = A
        self.B = B

        if applicable:
            print("Sweep method is applicable")
            print()

            tri_equation = equation.Equation(triA, B)
            y = tri_equation.sweep_method().to_list()
        else:
            print("Sweep method is not applicable as there is no diagonal prevalence")
            print("LU decomposition and two pass are used")
            print()

            simple_equation = equation.Equation(A, B)

            y = simple_equation.analytic_solution().to_list()

        if not der_a:
            y.insert(0, y_a)

        if not der_b:
            y.append(y_b)

        x = []
        pos = self.a
        while abs(pos - self.b) > eps:
            x.append(pos)
            pos += h
        x.append(pos)

        y_true = [self.f_true(el) for el in x]

        return pd.DataFrame(zip(x, y, y_true), columns=['x', 'y', 'true_y'])


class Volterra2:
    class OutOfSegmentError(Exception):
        def __str__(self):
            return "Following x_i is out of segment"

    def __init__(self,
                 K,
                 f,
                 A,
                 a,
                 b,
                 h=0.1,
                 eps=0.000000001):
        self.K = K
        self.f = f
        self.A = A
        self.a = a
        self.b = b
        self.h = h
        self.eps = eps

        self.y = [[f(a)]]
        self.i = 1
        self.N = int((self.b - self.a) / h)

    def __calc_c__(self, c, answer, j=0):
        if j == len(self.y):
            answer.append(c)
            return

        for y in self.y[j]:
            self.__calc_c__(
                    c + self.A(j) * self.K(
                        self.a + self.i * self.h,
                        self.a + j * self.h,
                        y
                    ),
                    answer,
                    j + 1
            )

    def next_equation(self):
        if self.i > self.N:
            raise Volterra2.OutOfSegmentError

        c = []
        x = self.a + self.i * self.h

        self.__calc_c__(
            self.f(x),
            c
        )

        self.i += 1

        return self.i - 1, x, self.A(self.i - 1), c

    # methods --- list of names
    # datas --- list of inputs (init_x + data for method)
    # init_x for dichotomy is tuple (a, b)
    # where a and b are line borders
    def solve_current(self, methods, datas):
        if self.i > self.N:
            raise Volterra2.OutOfSegmentError

        new_ys = []

        for method, data in zip(methods, datas):
            solver = NonLinear(data[0], self.eps)

            new_y = getattr(solver, method)(*data[1:]).answer

            if method == 'dichotomy':
                new_y = (new_y[0] + new_y[1]) / 2
            new_ys.append(new_y)

        self.y.append(new_ys)

    def solve_all(self, init_x, method):
        for i in range(1, self.N + 1):
            _, x, A, c = self.next_equation()

            self.solve_current(
                [method],
                [(
                    init_x,
                    lambda y: y[0] - A * self.K(x, x, y[0]) - c[0]
                )]
            )

    def __construct_solutions__(self, answer, con=None, i=0):
        if not con:
            con = []

        if i == len(self.y):
            answer.append(list(con))
            return

        for y in self.y[i]:
            con.append(y)
            self.__construct_solutions__(answer, con, i+1)
            con.pop()

    def get_solutions(self):
        y = []
        self.__construct_solutions__(y)
        return y
