import math
import pandas as pd

from pulp import *
import numpy as np
import random

def create_node_map(n):
    # Assumption: map shape is square
    n_side = int(math.sqrt(n))
    nmap = np.zeros((n_side, n_side))
    x = 0
    for i in range(n_side):
        for j in range(n_side):
            nmap[i][j] = x
            x += 1
    return nmap

def create_arc_weights(node_map, n, w_unit):
    wmap = np.zeros((n, n))
    for i in range(n):
        node_f = np.where(node_map == i)
        for j in range(n):
            node_s = np.where(node_map == j)
            if i == j:
                wmap[i][j] = 0
            else:
                wmap[i][j] = round(math.sqrt((node_f[0]-node_s[0])**2+(node_f[1]-node_s[1])**2)*w_unit)
    return wmap

def create_node_demand(weight_map, n, R, Q, demand_limit):
    n_side = int(math.sqrt(n))
    dmap = np.zeros((n_side, n_side))

    while True:
        for i in range(n_side):
            for j in range(n_side):
                if i == 0 and j == 0:
                    dmap[i][j] = 0
                else:
                    # dmap[i][j] = int(random.uniform(Q/10, np.floor(Q - 2 * np.max(weight_map))))
                    dmap[i][j] = int(random.uniform(demand_limit[0],demand_limit[1]))
        if sum(sum(dmap)) <= R*np.floor(Q - 2 * np.max(weight_map)):
            return True, dmap, dmap.flatten()
        else:
           # print("Cannot Cover")
            return False, 0, 0

def create_multi_weight_map(weight_map, R):
    multi_weight_map = weight_map
    for i in range(R-1):
        if i == 0:
            multi_weight_map = np.vstack((multi_weight_map[None], weight_map[None]))
        else:
            multi_weight_map = np.vstack((multi_weight_map, weight_map[None]))
    return multi_weight_map

def create_dataset(n, R, Q, w_unit, demand_limit):

    node_map = create_node_map(n)
    weight_map = create_arc_weights(node_map, n, w_unit)
    status, demand_map, demand = create_node_demand(weight_map, n, R, Q, demand_limit)
    if not status:
        return -1, "Not Cover"
    multi_weight_map = create_multi_weight_map(weight_map, R)

    node_list = list(range(0, n))
    node_list_demand = list(range(1, n))
    car_list = list(range(1, R+1))

    costs = makeDict([car_list, node_list, node_list], multi_weight_map, 0)

    # Solver
    model = LpProblem("mt-CCVRP", LpMinimize)

    x = LpVariable.dicts("Route", (car_list, node_list, node_list), 0, 1, cat="Binary")  # variable "x"

    routes = list(filter(None, [((r, i, j) if i != j else None) for r in car_list for i in node_list for j in node_list]))
    model += lpSum([x[r][i][j]*costs[r][i][j] for (r, i, j) in routes]), "Objective Function"  # Objective Function (1)

    for j in node_list_demand:
        parameter = list(filter(None, [((r, i, j) if i != j else None) for r in car_list for i in node_list ]))
        model += lpSum([x[r][i][j] for (r, i, j) in parameter]) == 1, "(2) Node %s Visit Constraint"%j  # Constraint (2)

    # for j in node_list_demand:
    #     model += lpSum([x[r][i][j] for r in car_list for i in node_list]) == 1, "(2) Node %s Visit Constraint"%j  # Constraint (2)

    for r in car_list:
        model += lpSum([x[r][0][j] for j in node_list_demand]) == 1, "(3) Vehicle %s Base Leave Constraint"%r  # Constraint (3)

    for r in car_list:
        for j in node_list:
            parameter = list(filter(None, [((r, i, j) if i != j else None) for i in node_list]))
            parameter_2 = list(filter(None, [((r, j, i) if i != j else None) for i in node_list]))
            model += lpSum([x[r][i][j] for (r, i, j) in parameter]) == lpSum([x[r][j][i] for (r, j, i) in parameter_2]), "(4) Vehicle %s Base Return From %s Constraint"%(r, j)  # Constraint (4)


    for r in car_list:
        parameter = list(filter(None, [((r, i, j) if i != j else None) for i in node_list for j in node_list_demand ]))
        model += lpSum([x[r][i][j]*(demand[j] + costs[r][i][j]) for (r, i, j) in parameter]) <= Q, "(5) Capacity of Vehicle %s Constraint"%r  # Constraint (5)

    # for r in car_list:
    #     model += lpSum([x[r][i][j]*(demand[j] + costs[r][i][j]) for i in node_list for j in node_list_demand]) <= Q, "(5) Capacity of Vehicle %s Constraint"%r  # Constraint (5)
    #
    parameter = list(filter(None, [((r, i, j) if i != j else None) for r in car_list for i in node_list_demand for j in node_list_demand]))
    model += lpSum([x[r][i][j] for (r, i, j) in parameter]) <= n-1+R, "(6) No Disconnect at the Routes Constraint"  # Constraint (6)

    # for r in car_list:
    #     model += lpSum([x[r][i][j] for i in node_list_demand for j in node_list_demand]) <= n-1, "(6) No Diconnect of Vehicle %s Constraint"%r  # Constraint (6)

    for r in car_list:
        for i in node_list:
            for j in node_list:
                model += lpSum([x[r][i][j]+x[r][j][i]]) <= 2, "(7) For Vehicle %s Not Use Same Road %s %s Constraint"%(r, i, j)  # Constraint (5)

    # print(model)
    # model.writeLP("CheckLpProgram.lp")

    status = model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=600, gapRel=0, threads=8))
    # status = model.solve(CPLEX_CMD(timeLimit=600, gapRel=0))
    # for var in model.variables():
    #     print(f"{var.name}: {var.value()}")
    # for name, constraint in model.constraints.items():
    #     print(f"{name}: {constraint.value()}")

    if model.status == 1:
        return model.objective.value(), LpStatus[model.status]
    else:
        return -1, LpStatus[model.status]


    # print(f"status: {model.status}, {LpStatus[model.status]}")
    # print(f"objective: {model.objective.value()}")


# variables
n_list = [9, 16, 25, 36, 49, 64, 81, 100]  # number of node
R_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # number of drones
Q_list = [180, 240, 300, 360, 420, 480, 540, 600]  # capacity of the drones
w_unit_list = [5, 10, 15, 20, 25, 30]  # unit weight of the arc
demand_limits = [60, 180]


for node in n_list:
    df = pd.DataFrame(columns=['node_number', 'drone_number', 'capacity_of_drones', 'arc_weight', 'result', 'type'])
    for drone in R_list:
        for capacity in Q_list:
            for weight_unit in w_unit_list:
                value, output_type = create_dataset(node, drone, capacity, weight_unit, demand_limits)
                row = {'node_number': node, 'drone_number': drone, 'capacity_of_drones': capacity, 'arc_weight': weight_unit, 'result': value, 'type': output_type}
                df = df.append(row, ignore_index=True)
                print('-----------------------------------------------------')
                print('Node {} Drone {} Capacity {} Weight {} Value {} Type {}'.format(node, drone, capacity, weight_unit, value, output_type))
                print('-----------------------------------------------------')
    df.to_csv('Output/output_{}.csv'.format(node))