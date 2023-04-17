# -*- coding: utf-8 -*-
"""
Assignment:
Metropolis algorithm II: Implement the traveling merchant problem.

Test data with solution can be found here:
http://www.math.uwaterloo.ca/tsp/
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

@author: Dennys Huber
"""

import matplotlib.pyplot as plt
import numpy as np
import random as rd
from matplotlib.animation import FuncAnimation
import networkx as nx


class Tour:
    def __init__(self, file_tour: str, file_opt: str) -> None:
        self.nr_cities = 0
        self.path = []
        self.ids = []
        self.opttour_ids = []
        self.temp = 1.0
        self.init_factor = 1000

        self.load_tsp(file_tour, file_opt)

        self.temp = 0
        self.length_path = length(self.path)

        self.start_temperature()
        self.minimum_energy_paths()

    def print(self):
        print("======= Tour =======")
        print("Length: ", self.length_path)
        print("nr_cities: ", self.nr_cities)
        print("path: \n", self.ids)
        print("optimal path:\n", self.opttour_ids)

    def load_tsp(self, coordinates: str, optimal_solution: str):
        file = coordinates
        x, y = np.loadtxt(
            file, delimiter=" ", comments="EOF", skiprows=6, usecols=(1, 2), unpack=True
        )

        path = []
        ids = []

        for i in range(len(x)):
            ids.append(i + 1)
            path.append([x[i], y[i]])

        file = optimal_solution
        opttour = np.loadtxt(
            file,
            delimiter=" ",
            comments="-1",
            dtype=int,
            skiprows=5,
            usecols=(0),
            unpack=True,
        )
        print("# data read: %i" % (len(opttour)))

        self.path = path
        self.nr_cities = len(path)
        self.ids = ids
        self.opttour_ids = opttour.tolist()

    def swap(self, idx, jdx):
        new_path = self.path.copy()
        new_ids = self.ids.copy()

        new_path[idx], new_path[jdx] = new_path[jdx], new_path[idx]
        new_ids[idx], new_ids[jdx] = new_ids[jdx], new_ids[idx]

        new_change = (
            distance(new_path[(idx - 1) % self.nr_cities], new_path[idx])
            + distance(new_path[idx], new_path[(idx + 1) % self.nr_cities])
            + distance(new_path[(jdx - 1) % self.nr_cities], new_path[jdx])
            + distance(new_path[jdx], new_path[(jdx + 1) % self.nr_cities])
        )
        old_change = (
            distance(self.path[(idx - 1) % self.nr_cities], self.path[idx])
            + distance(self.path[idx], self.path[(idx + 1) % self.nr_cities])
            + distance(self.path[(jdx - 1) % self.nr_cities], self.path[jdx])
            + distance(self.path[jdx], self.path[(jdx + 1) % self.nr_cities])
        )
        dist_change = new_change - old_change

        return self.length_path + dist_change, new_path, new_ids

    def reverse(self, idx, jdx):
        new_path = self.path.copy()
        new_ids = self.ids.copy()
        split = (1 + (jdx - idx) % self.nr_cities) / 2
        split = int(split)
        for i in range(split):
            start = (idx + i) % self.nr_cities
            end = (jdx - i) % self.nr_cities
            new_path[start], new_path[end] = new_path[end], new_path[start]
            new_ids[start], new_ids[end] = new_ids[end], new_ids[start]

        new_change = distance(
            new_path[(idx - 1) % self.nr_cities], new_path[idx]
        ) + distance(new_path[jdx], new_path[(jdx + 1) % self.nr_cities])
        old_change = distance(
            self.path[(idx - 1) % self.nr_cities], self.path[idx]
        ) + distance(self.path[jdx], self.path[(jdx + 1) % self.nr_cities])
        dist_change = new_change - old_change

        return self.length_path + dist_change, new_path, new_ids

    def random_move(self, idx, jdx):
        if rd.choice([True, False]):
            return self.swap(idx, jdx)
        else:
            return self.reverse(idx, jdx)

    def start_temperature(self):
        max_length_change = 0
        for _ in range(self.init_factor * self.nr_cities):
            idx, jdx = rd.randint(0, self.nr_cities - 1), rd.randint(
                0, self.nr_cities - 1
            )
            new_length, _, _ = self.random_move(idx, jdx)
            length_change = abs(self.length_path - new_length)
            max_length_change = max(length_change, max_length_change)
        self.temp = max_length_change
        print("INITIAL Temprature: ", self.temp)

    def minimum_energy_paths(self):
        min_length = self.length_path
        min_path = self.path
        print("INITIAL min length: ", min_length)
        for _ in range(self.nr_cities * self.init_factor):
            new_path = self.path.copy()
            np.random.default_rng().shuffle(new_path)
            new_length = length(new_path)
            if min_length > new_length:
                print("NEW min length: ", new_length)
                min_length = new_length
                min_path = new_path
        self.length_path = min_length
        self.path = min_path

    def metropolis(self):
        idx, jdx = rd.randint(0, self.nr_cities - 1), rd.randint(0, self.nr_cities - 1)
        length, new_path, new_ids = self.random_move(idx, jdx)
        if length < self.length_path:
            self.path = new_path
            self.length_path = length
            self.ids = new_ids
        else:
            float_random = rd.random()
            if float_random < np.exp((self.length_path - length) / self.temp):
                self.path = new_path
                self.length_path = length
                self.ids = new_ids


def distance(start, dest):
    return np.sqrt((dest[0] - start[0]) ** 2 + (dest[1] - start[1]) ** 2)


def length(path):
    dist = 0.0
    nr_cities = len(path)
    for i in range(nr_cities):
        if i > 0:
            ic = distance(path[i - 1], path[i])
        else:
            ic = distance(path[nr_cities - 1], path[i])
        dist += ic
    return dist


def monte_carlo_simulations(
    tour: Tour,
    nr_per_temperature: int,
    stopping_temperaturue: float,
    factor_temperature: float,
):
    tour_path_snapshots = []
    ids_snapshots = []
    counter = 0
    while tour.temp > stopping_temperaturue:

        for _ in range(nr_per_temperature):
            tour.metropolis()
        tour_path_snapshots.append(tour.path.copy())
        ids_snapshots.append(tour.ids.copy())
        counter += 1
        print("Iteration: ", counter)
        tour.temp *= factor_temperature
    return tour_path_snapshots, ids_snapshots


def plot_tour():
    G = nx.DiGraph()
    G.add_nodes_from(range(tour.nr_cities))
    positions = {}
    for i, pos in enumerate(tour.path):
        positions[i] = (pos[0], pos[1])

    node_size = 50

    frame = len(path_snapshot)

    def animate(frame):
        ax.clear()
        G.remove_edges_from(list(G.edges()))
        id_current = ids_snapshots[frame]
        len_ids = len(id_current)
        for id in range(len_ids):
            G.add_edge(id_current[id] - 1, id_current[(id + 1) % len_ids] - 1)

        G.add_edge(id_current[len_ids - 1] - 1, id_current[0] - 1)
        nx.draw(G, positions, with_labels=False, ax=ax, node_size=node_size)
        ax.set_title(f"Frame {frame}")

    am = FuncAnimation(fig, animate, frames=range(frame), interval=100, blit=False)
    # am.save("tsp.mp4", writer="ffmpeg", dpi=400)
    plt.show()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    TEMP_END = 0.1
    TEMP_FACTOR = 0.98
    NR_PER_TEMP = 1000

    tour = Tour("ch130.tsp", "ch130.opt.tour")

    path_snapshot, ids_snapshots = monte_carlo_simulations(
        tour, NR_PER_TEMP, TEMP_END, TEMP_FACTOR
    )

    tour.print()
    plot_tour()
