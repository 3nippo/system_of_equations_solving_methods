import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys

import math
import numpy as np
import itertools

import context
import matrix
from difeq.finite_differences.laplace import Laplace


error = 1e-6

omega = 0.001

h1 = 1e-1
h2 = 1e-1

l1 = math.pi/2
l2 = math.pi/2

A_coefs = [
    4 - 2 / h1**2 - 2 / h2**2,
    (h1 + 1) / h1**2,
    (-h1 + 1) / h1**2,
    (h2 + 1) / h2**2,
    (-h2 + 1) / h2**2
]

def analytic_solution(x, y):
    return math.exp(-x -y) * math.cos(x) * math.cos(y)

def phi1(y):
    return math.exp(-y) * math.cos(y)

def phi2(y):
    return 0

def psi1(x):
    return math.exp(-x) * math.cos(x)

def psi2(x):
    return 0

def get_n_steps(h1, h2):
    n_x_steps = int(l1 / h1) + 1
    n_t_steps = int(l2 / h2) + 1

    return n_x_steps, n_t_steps

def get_analytic_solution_on_grid(x_step, y_step):
    n_x_steps, n_y_steps = get_n_steps(x_step, y_step)

    solution = [analytic_solution(i * x_step, j * y_step) for j in range(n_y_steps) for i in range(n_x_steps)]

    return solution

def rmse(y, y_pred):
    y_pred = y_pred.to_list()
    
    sum = 0
    
    for i in range(len(y)):
        sum += (y[i] - y_pred[i])**2

    N1, N2 = get_n_steps(h1, h2)

    return math.sqrt(sum) / (N1 * N2)

def evaluate_method(name, *args):
    solver = Laplace(
        h1,
        h2,
        l1,
        l2,
        A_coefs,
        phi1,
        phi2,
        psi1,
        psi2,
        error
    )
    
    solution = getattr(solver, name)(*args)
    expected = get_analytic_solution_on_grid(h1, h2)
    
    err = rmse(expected, solution.answer)

    return err, solution.iterations

def error_fixed(args):
    ranged_first = isinstance(args[0], list)

    ranged = args[(int(ranged_first) + 1) % 2]
    
    ranged = sorted(ranged)

    if ranged_first:
        y_step = args[1]
    else:
        x_step = args[0]
    
    axis_name = 'x step' if ranged_first else 'y step'

    errors = []

    for el in ranged:
        if ranged_first:
            x_step = el
        else:
            y_step = el

        solver = Laplace(
            x_step,
            y_step,
            l1,
            l2,
            A_coefs,
            phi1,
            phi2,
            psi1,
            psi2,
            error
        )

        solution = solver.relax(omega)

        expected = get_analytic_solution_on_grid(x_step, y_step)
        
        err = rmse(expected, solution.answer)

        errors.append(err)

    _, ax = plt.subplots()
    
    length = l1 if ranged_first else l2

    ax.plot([length / el + 1 for el in ranged], errors)
    plt.xlabel(axis_name + 's')
    plt.ylabel('rmse error per point')

    plt.legend()

    return ax

names = [
    "simple",
    "zeydel",
    "relax"
]

if len(sys.argv) == 1:
    for i, name in enumerate(names):
        args = [name]

        if i == 2:
        # if i == 0:
            args.append(omega)
        
        err, iterations = evaluate_method(*args)

        print("{}\nerror: {}\niterations: {}\n".format(
            name, 
            err,
            iterations
        ))
else:
    ax1 = error_fixed(
        [
            [l1 / el for el in [10, 50, 100, 150]],
            0.1       
        ]
    )

    ax2 = error_fixed(
        [
            0.1,       
            [l2 / el for el in [10, 50, 100, 150]]
        ]
    )

plt.show()
