import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class City:
    def __init__(self,x, y) -> None:
        self.x = x 
        self.y = y
    
def distance(city_start, city_dest):
    return np.sqrt((city_start.x-city_dest.x)**2 + (city_start.y-city_dest.y)**2)

def total_distance(tour):
    dist = 0
    for i, j in zip(tour[:-1], tour[1:]):
        dist += distance(i,j)
    dist += distance(tour[0], tour[-1])
    return dist

def swap(tour, idx, jdx):
    tour[idx], tour[jdx] = tour[jdx], tour[idx] 
    return tour

def reverse(tour, idx, jdx):
    l = len(tour)
    split = (1 + (jdx - idx) % l) / 2
    split = int(split)
    for i in range(split):
        start = (idx + i) % l
        end = (jdx - i) % l
        tour[start], tour[end] = tour[end], tour[start]
    return tour

def init_temp(tour):
    max_length = total_distance(tour)
    max_change = 0
    for _ in range(1000):
        idx, jdx = np.random.randint(0, len(tour), size=2)
        new_tour = swap(tour.copy(), idx, jdx)
        new_length = total_distance(new_tour)
        if abs(max_length-new_length) > max_change:
            max_change= abs(max_length-new_length)
            max_length = new_length
            print("INITIAL Temprature: ", max_change)
    return max_change

def init_tour(tour):
    min_length = total_distance(tour)
    min_tour = tour.copy()
    print("INITIAL min length: ", min_length)
    for _ in range(1000):
        min_path = tour.copy()
        np.random.default_rng().shuffle(min_path)
        new_length = total_distance(min_path)
        if min_length > new_length:
            print("NEW min length: ", new_length)
            min_length = new_length
            min_tour = min_path
    return min_tour, min_length




def metropolis(tour):
    tour, length = init_tour(tour)

    temp = init_temp(tour)
    temp_factor = 0.98
    tour_snapshots = []

    for i in range(500):
        print(i, ' temperature = ', length)

        temp *= temp_factor

        for _ in range (1000):
            idx, jdx = np.random.randint(0, len(tour), size=2)
            tour = swap(tour, idx, jdx)

            new_length = total_distance(tour)

            if new_length < length:
                length = new_length;

            else:
                randy = np.random.uniform()
                if randy < np.exp((length - new_length)/temp):
                    length = new_length
                else:
                    tour = swap(tour, idx, jdx)

            idx, jdx = np.random.randint(0, len(tour), size=2)

            tour = reverse(tour, idx, jdx)

            new_length = total_distance(tour)

            if new_length < length:
                length = new_length;

            else:
                randy = np.random.uniform()
                if randy < np.exp((length - new_length)/temp):
                    length = new_length
                else:
                    tour = reverse(tour, idx, jdx)
        tour_snapshots.append(tour.copy())
    return tour_snapshots


def load_TSP(tour, opt):
    file = tour
    x, y = np.loadtxt(file, delimiter=' ', comments="EOF",
                      skiprows=6, usecols=(1, 2), unpack=True)

    print("# data read: %i" % (len(x)))
    cities = []
    for i in range(len(x)):
        cities.append(City(x[i], y[i]))

    file = opt
    opttour = np.loadtxt(file, delimiter=' ', comments="-1",
                      dtype=int, skiprows=5, usecols=(0), unpack=True)
    print("# data read: %i" % (len(opttour)))
    print("Optimal tour\n", opttour)

    return cities

def plot_TSP():
    line, = plt.plot([], [], lw=2)
    def init():
        x = [c.x for c in snapshots[0]]
        x.append(snapshots[0][0].x)
        y = [c.y for c in snapshots[0]]
        y.append(snapshots[0][0].y)
        plt.plot(x, y, 'ro')

        extra_x = (max(x) - min(x)) * 0.05
        extra_y = (max(y) - min(y)) * 0.05
        ax.set_xlim(min(x) - extra_x, max(x) + extra_x)
        ax.set_ylim(min(y) - extra_y, max(y) + extra_y)

        line.set_data([], [])
        return line,

    def update(frame):
        x = [c.x for c in snapshots[frame]]
        x.append(snapshots[frame][0].x)
        y = [c.y for c in snapshots[frame]]
        y.append(snapshots[frame][0].y)
        line.set_data(x, y)
        return line


    am = FuncAnimation(fig, update, frames=range(len(snapshots)),
                        init_func=init, interval=50, repeat=False)

    am.save("tsp2.mp4", writer="ffmpeg", dpi=400)
    plt.show()
 
if __name__ == "__main__":
    tour = []

    tour = load_TSP("./ch130.tsp", "./ch130.opt.tour")

    fig, ax = plt.subplots()

    snapshots = metropolis(tour)

    plot_TSP()





