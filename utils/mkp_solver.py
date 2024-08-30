import numpy as np
from pyscipopt import Model, quicksum, multidict


def mkp(I, J, v, a, b):
    """mkp -- model for solving the multi-constrained knapsack
    Parameters:
        - I: set of dimensions
        - J: set of items
        - v[j]: value of item j
        - a[i,j]: weight of item j on dimension i
        - b[i]: capacity of knapsack on dimension i
    Returns a model, ready to be solved.
    """
    model = Model("mkp")

    # Create Variables
    x = {}
    for j in J:
        x[j] = model.addVar(vtype="B", name="x(%s)" % j)

    # Create constraints
    for i in I:
        model.addCons(quicksum(a[i, j] * x[j] for j in J) <= b[i], "Capacity(%s)" % i)

    # Objective
    model.setObjective(quicksum(v[j] * x[j] for j in J), "maximize")
    model.data = x
    model.hideOutput(quiet=True)

    return model


def mkp_interface(state, bw, mem, comput, bw_max, mem_max, comput_max):
    """mkp -- model for solving the multi-constrained knapsack
    Parameters:
        - I: set of dimensions
        - J: set of items
        - v[j]: value of item j
        - a[i,j]: weight of item j on dimension i
        - b[i]: capacity of knapsack on dimension i
    Returns a model, ready to be solved.
    """
    items = dict()
    const = dict()
    for idx in range(len(state)):
        items[idx] = state[idx]
        const[(1,idx)] = bw[idx]
        const[(2, idx)] = mem[idx]
        const[(3, idx)] = comput[idx]
    J, v = multidict(items)
    a = const
    I, b = multidict({1: bw_max, 2: mem_max, 3: comput_max})

    sol = np.array(mkp_solver(I, J, v, a, b))
    for idx in range(len(state)):
        if state[idx] == 0:
            sol[idx] = 0

    return sol


def mkp_solver(I, J, v, a, b):
    model = mkp(I, J, v, a, b)
    x = model.data
    model.optimize()

    sol = []
    for i in x:
        v = x[i]
        sol.append(model.getVal(v))
    return sol