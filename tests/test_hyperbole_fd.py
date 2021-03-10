import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys

import math
import numpy as np
import itertools

import context
import matrix
from difeq import finite_differences


a_squared=1e-2
a=math.sqrt(a_squared)
x_step=1e-2
t_step=1e-2
l=math.pi
T=2

def analytic_solution(x, t):
    return math.sin(x - a * t)

def phi_start(t):
    return -math.sin(a * t)

def phi_end(t):
    return math.sin(l - a * t)

def psi_1(x):
    return math.sin(x)

def psi_2(x):
    return -a * math.cos(x)

def psi_1_sec_der(x):
    return -math.sin(x)

def get_n_steps(x_step, t_step):
    n_x_steps = int(l / x_step) + 1
    n_t_steps = int(T / t_step) + 1

    return n_x_steps, n_t_steps

def get_analytic_solution_on_grid(x_step, t_step):
    n_x_steps, n_t_steps = get_n_steps(x_step, t_step)

    solution = [[analytic_solution(j * x_step, k * t_step) for j in range(n_x_steps)] for k in range(n_t_steps)]

    return solution

def evaluate_method(name):
    solver = finite_differences.Hyperbole(
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
    )

    solution = getattr(solver, name)()
    expected = get_analytic_solution_on_grid(x_step, t_step)

    n_x_steps, n_t_steps = get_n_steps(x_step, t_step)
    
    fig, ax = plt.subplots(2, 1)
    
    x_range = [j * x_step for j in range(n_x_steps)]
    
    fig.suptitle(name, fontsize=14)

    ax[0].plot(x_range, solution[n_t_steps // 2], label='numeric')
    ax[0].plot(x_range, expected[n_t_steps // 2], label='analytic')
    ax[0].set_title('1/2 of the way')
    
    ax[1].plot(x_range, solution[n_t_steps - 1], label='numeric')
    ax[1].plot(x_range, expected[n_t_steps - 1], label='analytic')
    ax[1].set_title('end of the way')

    plt.legend()

    return ax

def error_fixed(name, args):
    ranged_first = isinstance(args[0], list)

    ranged = args[(int(ranged_first) + 1) % 2]
    
    ranged = sorted(ranged)

    if ranged_first:
        t_step = args[1]
    else:
        x_step = args[0]
    
    # def flatten(x):
    #     return list(itertools.chain.from_iterable(x))
    
    fig, ax = plt.subplots()

    axis_name = 'x step' if ranged_first else 'time step'

    errors = []

    for el in ranged:
        if ranged_first:
            x_step = el
        else:
            t_step = el

        solver = finite_differences.Hyperbole(
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
        )

        solution = getattr(solver, name)()
        solution = matrix.Matrix.from_2d(solution) 
        
        expected = get_analytic_solution_on_grid(x_step, t_step)
        expected = matrix.Matrix.from_2d(expected) 
        
        err = (expected - solution).abs().max(axis=0).to_list()
        
        errors.append(max(err))

        n_x_steps, n_t_steps = get_n_steps(x_step, t_step)

        time_steps = [i for i in range(get_n_steps(x_step, t_step)[1])]
        
        ax.plot(
            time_steps[1:], 
            err[1:], 
            label='{} = {}'.format(
            axis_name, 
            el
            )
        )

    plt.xlabel('time steps')
    plt.ylabel('error')

    plt.legend()

    _, ax2 = plt.subplots()
    
    length = l if ranged_first else T

    ax2.plot([length / el + 1 for el in ranged], errors)
    plt.xlabel(axis_name + 's')
    plt.ylabel('max error from time step')

    return ax, ax2

names = ["implicit", "explicit"]

for i, name in enumerate(names):
    print(i+1, name)

name_index = int(input())

name = names[name_index - 1]

if len(sys.argv) == 1:
    ax1 = evaluate_method('{}'.format(name))
    ax2 = evaluate_method('{}'.format(name))
    ax3 = evaluate_method('{}'.format(name))

else:
    ax1, ax2 = error_fixed(
        '{}'.format(name), 
        [
            [l / el for el in [10, 50, 100, 150, 200, 250, 300, 350, 400]],
            0.001       
        ]
    )

    ax3, ax4 = error_fixed(
        '{}'.format(name), 
        [
            0.001,       
            [T / el for el in [10, 50, 100, 150, 200, 250, 300, 350, 400]]
        ]
    )

plt.show()
