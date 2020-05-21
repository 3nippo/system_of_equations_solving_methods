import pandas as pd


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
