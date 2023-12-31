import pandas as pd

from ortools.linear_solver import pywraplp


# Define input data
prices = pd.DataFrame(
    index=["January", "February", "March", "April", "May", "June"],
    data={
        "VEG 1": [110, 130, 110, 120, 100,  90],
        "VEG 2": [120, 130, 140, 110, 120, 100],
        "OIL 1": [130, 110, 130, 120, 150, 140],
        "OIL 2": [110,  90, 100, 120, 110,  80],
        "OIL 3": [115, 115,  95, 125, 105, 135],
    },
)
hardness = [8.8, 6.1, 2.0, 4.2, 5.0]
price_per_ton = 150
storage_cost_per_ton_per_month = 5
storage_limit = 1000
hardness_lower_bound = 3
hardness_upper_bound = 6
initial_storage_amount = 500
final_storage_amount = 500
maximum_vegetable_refinement_per_month = 200
maximum_non_vegetable_refinement_per_month = 250


if __name__ == "__main__":

    # Create solver
    solver = pywraplp.Solver.CreateSolver("GLOP")

    # Index sets
    I = range(0, prices.shape[1])  # oils
    J = range(0, prices.shape[0])  # months

    # Upper bounds
    upper_bound_x = max(maximum_vegetable_refinement_per_month, maximum_non_vegetable_refinement_per_month)
    upper_bound_b = storage_limit + upper_bound_x
    upper_bound_y = maximum_vegetable_refinement_per_month + maximum_non_vegetable_refinement_per_month

    # Define variables
    x = [[solver.NumVar(0, upper_bound_x, f"x_{i}{j}") for j in J] for i in I]  # amount of oil i to be refined in month j
    b = [[solver.NumVar(0, upper_bound_b, f"b_{i}{j}") for j in J] for i in I]  # amount of oil i to be bought in month j
    s = [[solver.NumVar(0, storage_limit, f"s_{i}{j}") for j in J] for i in I]  # amount of oil i in storage at the end of month j
    y = [solver.NumVar(0, upper_bound_y, f"y_{j}") for j in J]  # production in month j
    print("Number of variables =", solver.NumVariables())

    # At most 200 tons of vegetable oils refined per month
    for j in J:
        solver.Add(x[0][j] + x[1][j] <= maximum_vegetable_refinement_per_month)

    # At most 250 tons of non-vegetable oils refined per month
    for j in J:
        solver.Add(x[2][j] + x[3][j] + x[4][j] <= maximum_non_vegetable_refinement_per_month)

    # Amount of product produced in month j
    for j in J:
        solver.Add(sum(x[i][j] for i in I) == y[j])

    # Hardness lower bound fulfilled
    for j in J:
        solver.Add(sum(hardness[i] * x[i][j] for i in I) >= hardness_lower_bound * y[j])

    # Hardness upper bound fulfilled
    for j in J:
        solver.Add(sum(hardness[i] * x[i][j] for i in I) <= hardness_upper_bound * y[j])

    # Exactly 500 of each type remaining at the end of june
    for i in I:
        solver.Add(s[i][5] == final_storage_amount)

    # Storage at the end of a month is consistent with the initial storage, the amount bought and the amount refined
    for i in I:
        for j in J:
            if j > 0:
                solver.Add(s[i][j] == s[i][j-1] + b[i][j] - x[i][j])

    # Storage at the end of January is consistent with initial storage of 500 of each type
    for i in I:
        solver.Add(s[i][0] == initial_storage_amount + b[i][0] - x[i][0])

    # Objective function
    material_costs = sum(prices.iloc[j].iloc[i] * b[i][j] for i in I for j in J)
    storage_costs = storage_cost_per_ton_per_month * sum(s[i][j] for i in I for j in J)
    revenues = price_per_ton * sum(y[j] for j in J)
    objective = revenues - material_costs - storage_costs
    solver.Maximize(objective)

    # Solve the program
    status = solver.Solve()

    # Check if a solution was found
    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        for i in I:
            for j in J:
                print(f'x_{i}{j} =', x[i][j].solution_value())
    else:
        print('The problem does not have an optimal solution.')
