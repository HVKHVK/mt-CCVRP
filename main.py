import math
from pulp import *
import numpy as np
import random

# variables
n = 25  # number of node
R = 3  # number of drones
Q = 300  # capacity of the drones
w_unit = 5  # unit weight of the arc

def create_node_map():
    # Assumption: map shape is square
    n_side = int(math.sqrt(n))
    nmap = np.zeros((n_side, n_side))
    x = 0
    for i in range(n_side):
        for j in range(n_side):
            nmap[i][j] = x
            x += 1
    return nmap

def create_arc_weights(node_map):
    wmap = np.zeros((n, n))
    for i in range(n):
        node_f = np.where(node_map == i)
        for j in range(n):
            node_s = np.where(node_map == j)
            if i == j:
                wmap[i][j] = 0
            else:
                wmap[i][j] = math.sqrt((node_f[0]-node_s[0])**2+(node_f[1]-node_s[1])**2)*w_unit
    return wmap

def create_node_demand(weight_map):
    n_side = int(math.sqrt(n))
    dmap = np.zeros((n_side, n_side))

    while True:
        for i in range(n_side):
            for j in range(n_side):
                if i == 0 and j == 0:
                    dmap[i][j] = 0
                else:
                    dmap[i][j] = int(random.uniform(0, np.floor(Q - 2 * np.max(weight_map))))
        if sum(sum(dmap)) >= R*np.floor(Q - 2 * np.max(weight_map)):
            return dmap

node_map = create_node_map()
weight_map = create_arc_weights(node_map)
demand_map = create_node_demand(weight_map)
x = np.zeros((R, n, n))  # Route enable variable

