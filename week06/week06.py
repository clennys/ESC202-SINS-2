# -*- coding: utf-8 -*-
"""
Assignment:
Using the Metropolis algorithm, plot the mean magnetization of a N by N grid of spins (+1 and -1) depending on the temperature. 
Visualize the spin state of the grid at different temperatures.

Use the following parameters as a starting point:
NX = 150
NY = 150
J = 1
N_PER_TEMP = 40 * NX * NY
TEMP_START = 4
TEMP_END = 0.1
TEMP_FACTOR = 0.98

 https://numba.pydata.org/numba-doc/latest/user/5minguide.html


@author: Dennys Huber
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import numpy.random as rd
from numba import njit


def grid_random_spins(width: int, height: int):
    grid = np.zeros((width, height))
    for i in range(height):
        for j in range(width):
            grid[i][j] = rd.choice([1,-1])
    return grid

@njit
def energy_change_spin(grid, idx: int, jdx:int, width:int, height:int, interaction_energy: float):
    center = grid[idx, jdx] 
    up = grid[idx, (jdx + 1) % height]
    right = grid[(idx + 1) % width, jdx]
    down= grid[idx, (jdx - 1) % height] 
    left = grid[(idx - 1) % width, jdx] 
    neighbors = up + right + down + left
    return 2 * interaction_energy * center * neighbors


@njit
def metropolis(grid, inv_temperature: float, width: int, height: int, interaction_energy: float):
    idx, jdx = rd.randint(0, width), rd.randint(0, height)
    energy_change = energy_change_spin(grid, idx, jdx, width, height, interaction_energy)
    if energy_change < 0:
        grid[idx, jdx] = -grid[idx, jdx]
    else:
        float_random = rd.random()
        if float_random < np.exp(-inv_temperature * energy_change):
            grid[idx, jdx] = -grid[idx, jdx]

def monte_carlo_simulations(grid, width: int, height: int, current_temperature: float, nr_per_temperature:int, stopping_temperaturue: float, factor_temperature: float, interaction_energy:float ):
    grid_snapshots = []
    counter = 0
    while current_temperature > stopping_temperaturue:
        current_temperature *= factor_temperature
        inv_temperature = 1/current_temperature # K_B const = 1

        for _ in range(nr_per_temperature):
            metropolis(grid, inv_temperature, width, height, interaction_energy)
        grid_snapshots.append(grid.copy())
        counter += 1
        print("Iteration: ", counter)
        # print(grid)
    return grid_snapshots


if __name__ == "__main__":
    fig, ax = plt.subplots()
    NX = 150
    NY = 150
    J = 1.0
    N_PER_TEMP = 40 * NX * NY
    TEMP_START = 4.0
    TEMP_END = 0.1
    TEMP_FACTOR = 0.98

    grid = grid_random_spins(NX, NY)

    grid_snapshots = monte_carlo_simulations(grid, NX, NY, TEMP_START, N_PER_TEMP, TEMP_END, TEMP_FACTOR, J)
    print(len(grid_snapshots))


    def update(frame):
        line.set_data(grid_snapshots[frame])
        return line,

    line = ax.imshow(grid_snapshots[0], cmap="winter", animated=True)

    frame = len(grid_snapshots)

    am = FuncAnimation(fig, update, frames=range(frame), blit=True)

    am.save("2d-ising-model.mp4", writer="ffmpeg", dpi=400)
    plt.show()
