import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys

import math
import numpy as np
import itertools

import context
import matrix
from difeq.finite_differences import ParabolaDim2


a = 1
b = 1
mu = 1

h1 = 1e-1
h2 = 1e-1
dt = 1e-1

l1 = math.pi
l2 = math.pi
T  = 1

def analytic_solution(x, y, t):
    return math.sin(x) * math.sin(y) * math.sin(mu * t)

def phi1(x, t):
    return 0

def phi2(x, t):
    return -math.sin(x) * math.sin(mu * t)

def psi1(y, t):
    return 0

def psi2(y, t):
    return -math.sin(y) * math.sin(mu * t)

def ksi(x, y):
    return 0

def f(x, y, t):
    return math.sin(x) * math.sin(y) * (mu * math.cos(mu * t) + (a + b) * math.sin(mu * t))

def get_n_steps(h1, h2, dt):
    n_x_steps = int(l1 / h1) + 1
    n_y_steps = int(l2 / h2) + 1
    n_t_steps = int(T / dt) + 1

    return n_x_steps, n_y_steps, n_t_steps

def get_analytic_solution_on_grid(h1, h2, dt):
    n_x_steps, n_y_steps, n_t_steps = get_n_steps(h1, h2, dt)
    
    solution = [[[analytic_solution(h1 * i, h2 * j, dt * k) for j in range(n_y_steps)] for i in range(n_x_steps)] for k in range(n_t_steps)]

    return solution

def rmse(y, y_pred, h1, h2, dt):
    def flatten(l):
        result = []

        for el in l:
            if isinstance(el, list):
                result.extend(flatten(el))
            else:
                result.append(el)

        return result
    
    y = flatten(y)
    y_pred = flatten(y_pred)
    
    sum = 0
    
    for i in range(len(y)):
        sum += (y[i] - y_pred[i])**2

    N1, N2, NT = get_n_steps(h1, h2, dt)

    return math.sqrt(sum) / (N1 * N2 * NT)

def evaluate_method(name):
    solver = ParabolaDim2(
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
    )
    
    solution = getattr(solver, name)()
    expected = get_analytic_solution_on_grid(h1, h2, dt)
    
    err = rmse(expected, solution, h1, h2, dt)

    return err

def error_fixed(name, args):
    for i, el in enumerate(args):
        if isinstance(el, list):
            ranged_index = i

    ranged = args[ranged_index]
    
    ranged = sorted(ranged)
    
    loc_args = [None] * len(args)

    for i in range(len(args)):
        if i != ranged_index:
            loc_args[i] = args[i]
    
    if ranged_index == 0:
        axis_name = 'x step'
    elif ranged_index == 1:
        axis_name = 'y step'
    else:
        axis_name = 't step'

    errors = []

    for el in ranged:
        loc_args[ranged_index] = el
        
        solver = ParabolaDim2(
            a,
            b,
            loc_args[0],
            loc_args[1],
            loc_args[2],
            l1,
            l2,
            T,
            phi1,
            phi2,
            psi1,
            psi2,
            ksi,
            f
        )

        solution = getattr(solver, name)()

        expected = get_analytic_solution_on_grid(*loc_args)
        
        err = rmse(expected, solution, *loc_args)

        errors.append(err)

    _, ax = plt.subplots()
    
    if ranged_index == 0:
        length = l1
    elif ranged_index == 1:
        length = l2
    else:
        length = T

    ax.plot([length / el + 1 for el in ranged], errors)
    plt.xlabel(axis_name + 's')
    plt.ylabel('rmse error per point from time step')

    plt.legend()

    return ax

names = [
    "variable_direction",
    "fractional_steps"
]

if len(sys.argv) == 1:
    for name in names:
        err = evaluate_method(name)

        print("{} error: {}\n".format(
            name, 
            err
        ))
else:
    for i, name in enumerate(names):
        print("{} {}".format(i+1, name))

    i = int(input())

    name = names[i-1]

    ax1 = error_fixed(
        name,
        [
            [l1 / el for el in [10, 50, 100, 150, 200, 250, 300, 350, 400, 700]],
            0.1,
            0.1
        ]
    )

    ax2 = error_fixed(
        name,
        [
            0.1,       
            [l2 / el for el in [10, 50, 100, 150, 200, 250, 300, 350, 400, 700]],
            0.1
        ]
    )

    ax3 = error_fixed(
        name,
        [
            0.1,
            0.1,
            [T / el for el in [10, 50, 100, 150, 200, 250, 300, 350, 400, 700]]
        ]
    )

plt.show()
