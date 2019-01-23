import numpy as np
from HarmonySearch import HarmonySearch as HS


# Minimize function
def default(args):
    x, y, z = args
    return (x - 2) ** 2 + (y - 3) ** 2 + (z - 1) ** 2 + 3


# Minimize function
def rosenbrock(args):
    x, y = args
    a = 1
    b = 100
    return (a - x) ** 2 + b * (y - x ** 2) ** 2


def test_params(func, dimension, global_min):
    for i in np.arange(0, 1, 0.1):
        for j in np.arange(0, 1, 0.1):
            hs = HS(N=dimension, M=10, HMCR=0.60, PAR=0.3, max_iter=100)
            hs.cost_func = func
            hs.solve()
            if hs.best_cost() == global_min:
                print('HMCR:', round(i, 1), 'PAR:', round(j, 1), '\t', 'Found global MIN')


def main(func, dimension, end_condition):
    hs = HS(N=dimension, M=10, HMCR=0.90, PAR=0.4, L=(0, 9), max_iter=100,
            debug=False, end_condition=end_condition)
    hs.cost_func = func
    hs.solve()
    hs.results()
    hs.plot_results()


if __name__ == '__main__':
    main(default, 3, 3)
    #test_params(rosenbrock, 2, 0)
