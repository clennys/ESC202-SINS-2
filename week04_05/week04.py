# -*- coding: utf-8 -*-
"""

@author Dennys Huber
"""

from logging import raiseExceptions
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import heapq as hq
from matplotlib.animation import FuncAnimation


class particle:
    def __init__(self, r, k):
        self.r = r  # position of the particle [x,y]
        self.rho = 0.0  # density of the particle
        self.mass = 1.0
        self.velo = np.zeros(2)
        self.predicted_velo = np.zeros(2)
        self.acc = np.zeros(2)
        self.energy = 1
        self.energy_dt = 0.0
        self.predicted_energy = 0.0
        self.c_sound = 0.0
        self.radius_of_kernel = 0.0
        self.k_nearest = k
        self.prio_q = None

    def __repr__(self):
        return (
            "(r: (x: %s, y: %s), rho: %s, m: %s, v: ((vx: %s, vy: %s), (vpx: %s, vpy: %s)), a: (ax: %s, ay: %s), e: (e: %s, ep: %s, edot: %s), c: %s, h: %s)"
            % (
                self.r[0],
                self.r[1],
                self.rho,
                self.mass,
                self.velo[0],
                self.velo[1],
                self.predicted_velo[0],
                self.predicted_velo[1],
                self.acc[0],
                self.acc[1],
                self.energy,
                self.predicted_energy,
                self.energy_dt,
                self.c_sound,
                self.radius_of_kernel,
            )
        )


class cell:
    def __init__(self, rLow, rHigh, lower, upper, offset):
        self.rLow = rLow  # [xMin, yMin]
        self.rHigh = rHigh  # [xMax, yMax]
        self.iLower = lower  # index to first particle in particle array
        self.iUpper = upper  # index to last particle in particle array
        self.pLower = None  # reference to tree cell for lower part
        self.pUpper = None  # reference to tree cell for upper part
        self.rc = self.center(rLow, rHigh)
        self.iOffset = offset

    def center(self, rLow, rHigh):
        return [(rLow[0] + rHigh[0]) / 2, (rLow[1] + rHigh[1]) / 2]

    def celldist2(self, r):
        """Calculates the squared minimum distance between a particle
        position and this node."""
        d1 = r - self.rHigh
        d2 = self.rLow - r
        d1 = np.maximum(d1, d2)
        d1 = np.maximum(d1, np.zeros_like(d1))
        return d1.dot(d1)


class prioqueue:
    def __init__(self, k):
        self.heap = []
        sentinel = (-np.inf, None)
        for _ in range(k):
            hq.heappush(self.heap, sentinel)

    def key(self):
        return -self.heap[0][0]

    def replace(self, dist, pos_in_array, r_without_offset):
        hq.heapreplace(self.heap, (-dist, pos_in_array, r_without_offset))

    def position_in_array(self, idx):
        return self.heap[idx][1]

    def __repr__(self):
        return str(self.heap)


def partition(A, i, j, v, d):
    """
    A: array of all particles
    i: start index of subarray for partition
    j: end index (inclusive) for subarray for partition
    v: value for partition e.g. 0.5
    d: dimension to use for partition, 0 (for x) or 1 (for y)
    """
    subA = A[i : j + 1]
    start, end = 0, len(subA) - 1
    while start <= end:
        if subA[start].r[d] < v and subA[end].r[d] >= v:
            start += 1
            end -= 1
        elif subA[start].r[d] >= v and subA[end].r[d] >= v:
            end -= 1
        elif subA[start].r[d] < v and subA[end].r[d] < v:
            start += 1
        elif subA[start].r[d] >= v and subA[end].r[d] < v:
            subA[start], subA[end] = subA[end], subA[start]
            start += 1
            end -= 1
    return start


def treebuild(A, root, dim):

    v = 0.5 * (root.rLow[dim] + root.rHigh[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)

    # print(A[0])

    if not s:
        return

    if s != 0:
        new_rHigh = root.rHigh[:]
        new_rHigh[dim] = v
        cLow = cell(root.rLow, new_rHigh, root.iLower, s - 1, root.iOffset)
        root.pLower = cLow
        if len(A[:s]) > 8:
            treebuild(A[:s], cLow, 1 - dim)

    if s <= len(A):
        new_rLow = root.rLow[:]
        new_rLow[dim] = v
        cHigh = cell(new_rLow, root.rHigh, 0, root.iUpper - s, root.iOffset + s)
        root.pUpper = cHigh
        if len(A[s:]) > 8:
            treebuild(A[s:], cHigh, 1 - dim)


def random_AMatrix(nr_particles, k_nearest):
    P = []
    A = np.array([])
    for _ in range(nr_particles):
        p = [rd.random(), rd.random()]
        P.append(p)

    P = np.asarray(P)
    P = P[P[:, 0].argsort()]

    for p in P:
        a = np.array([particle([p[0], p[1]], k_nearest)])
        A = np.append(A, a)
    return A


def plottree(root):
    # draw a rectangle specified by rLow and rHigh
    if root.pLower:
        plottree(root.pLower)
    if root.pUpper:
        plottree(root.pUpper)
    xl, xh = root.rLow[0], root.rHigh[0]
    yl, yh = root.rLow[1], root.rHigh[1]
    plt.plot([xl, xh], [yl, yl], color="red")
    plt.plot([xl, xh], [yh, yh], color="red")
    plt.plot([xl, xl], [yl, yh], color="blue")
    plt.plot([xh, xh], [yl, yh], color="blue")


def neighbor_search_periodic(pq, root, particles, r, period):
    # walk the closest image first (at offset=[0, 0])
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            rOffset = np.array([x, y])
            neighbor_search(pq, root, particles, r, rOffset)


def dist2(rc, ri):
    d2 = 0
    for d in range(2):
        d2 += (rc[d] - ri[d]) ** 2
    return d2


def neighbor_search(pq, root, particles, r, rOffset):
    """Do a nearest neighbor search for particle at  'r' in the tree 'root'
    using the priority queue 'pq'. 'rOffset' is the offset of the root
    node from unit cell, used for periodic boundaries.
    'particles' is the array of all particles.
    """
    if root is None:
        return

    ri = r + rOffset
    if root.pLower is not None and root.pUpper is not None:
        d2_lower = dist2(root.pLower.rc, ri)
        d2_upper = dist2(root.pUpper.rc, ri)
        if d2_lower <= d2_upper:
            if root.pLower.celldist2(ri) < pq.key():
                neighbor_search(pq, root.pLower, particles, r, rOffset)
            if root.pUpper.celldist2(ri) < pq.key():
                neighbor_search(pq, root.pUpper, particles, r, rOffset)
        else:
            if root.pUpper.celldist2(ri) < pq.key():
                neighbor_search(pq, root.pUpper, particles, r, rOffset)
            if root.pLower.celldist2(ri) < pq.key():
                neighbor_search(pq, root.pLower, particles, r, rOffset)
    elif root.pLower is not None:
        neighbor_search(pq, root.pLower, particles, r, rOffset)
    elif root.pUpper is not None:
        neighbor_search(pq, root.pUpper, particles, r, rOffset)
    else:  # root is a leaf cell
        # print("LEAF")
        for j in range(root.iLower + root.iOffset, root.iUpper + root.iOffset + 1):
            d2 = dist2(particles[j].r, ri)
            # print(particles[j].r, j, d2)
            if d2 < pq.key() and d2 != 0:
                pq.replace(d2, j, particles[j].r - rOffset)


def check_prio(k, A):
    # O(N**2) Test Code
    # k = Number of nearest neighbors
    for p in A:
        NN = []
        d2NN = []
        randy = []
        for i in range(k):
            d2min = float("inf")
            for q in A:
                if p != q and q not in NN:
                    d2 = dist2(p.r, q.r)
                    if d2 < d2min:
                        d2min = d2
                        qmin = q
            if p == A[rd_particle_idx]:
                randy.append([d2min, qmin])
            NN.append(qmin)
            d2NN.append(d2min)
            if len(randy) == k:
                return randy
    return []
    # Here NN and d2NN lists for particle p are filled.
    # Compare them with the lists you got from the recursive algorithm


def top_hat_kernel(radius, max_dist):
    return 1 / (max_dist**2 * np.pi) if 0 <= radius / max_dist <= 1 else 0


def monaghan_kernel(radius, max_dist):
    factor = (40 / (7 * np.pi)) / (max_dist**2)
    ratio = radius / max_dist

    if 0.0 <= ratio < 0.5:
        return factor * (6 * (ratio**3 - ratio**2) + 1)
    elif 0.5 <= radius / max_dist <= 1:
        return factor * 2 * (1 - ratio) ** 3
    else:
        return 0


def monaghan_derivative(r, h):
    if r / h < 0.5:
        return 3 * (r / h) ** 2 - 2 * (r / h)
    elif 0.5 <= r / h <= 1:
        return -((1 - (r / h)) ** 2)
    return 0


def density(kernel, A, prio_q):
    total_rho = 0
    max_dist = np.sqrt(prio_q.key())

    for p in prio_q.heap:
        total_rho += A[p[1]].mass * kernel(np.sqrt(-p[0]), max_dist)

    return total_rho


def plot_density(method, k_nearest, A, root):
    densities = []
    x = []
    y = []

    for particle in A:
        prio_q = prioqueue(k_nearest)
        neighbor_search_periodic(prio_q, root, A, particle.r, np.array([1, 1]))
        rho = density(method, A, prio_q)
        particle.rho = rho
        densities.append(rho)
        x.append(particle.r[0])
        y.append(particle.r[1])

    return densities, x, y


def first_drift(A, time_step):
    for particle in A:
        for d in range(2):
            # print(particle.predicted_velo[d], particle.velo[d], particle.acc[d], time_step)
            particle.r[d] += particle.velo[d] * time_step
            particle.r[d] = particle.r[d] % 1.0
            particle.predicted_velo[d] = particle.velo[d] + particle.acc[d] * time_step
        particle.predicted_energy = particle.energy + particle.energy_dt * time_step


def kick(A, time_step):
    for particle in A:
        for d in range(2):
            # print("VELO: ",d, particle.acc[d], time_step)
            particle.velo[d] += particle.acc[d] * time_step
        particle.energy += particle.energy_dt * time_step


def second_drift(A, time_step):
    for particle in A:
        for d in range(2):
            particle.r[d] += particle.velo[d] * time_step
            particle.r[d] = particle.r[d] % 1


def gradient_monaghan_kernel(normal_r, radius, kernel_size_h):
    # print(((40 * 6 / (7 * np.pi)), " / ", kernel_size_h**3) , " * ", monaghan_derivative(radius, kernel_size_h), " = ", ((40 * 6 / (7 * np.pi)) / kernel_size_h**3) * monaghan_derivative(radius, kernel_size_h), " r: ", radius, " kernel: ", kernel_size_h, " normal: ", normal_r)
    dW = ((40 * 6 / (7 * np.pi)) / kernel_size_h**3) * monaghan_derivative(
        radius, kernel_size_h
    )
    return dW * normal_r


def r_diff(particle_a, particle_b):
    dist_particle_ab = np.zeros(2)
    for d in range(2):
        dist_particle_ab[d] = particle_a.r[d] - particle_b.r[d]
    return dist_particle_ab


def wendtland_kernel(abs_r, max_radius_h):
    h = max_radius_h / 2
    q = abs_r / h
    factor_alpha = 7 / (4 * np.pi * h**2)
    if 0.0 <= q <= 2.0:
        return factor_alpha * (1 - q / 2) ** 4 * (1 + 2 * q)
    elif 2 < q:
        return 0
    else:
        raise Exception("Wendland_kernel smth is wrong here")


def gradient_wendlant_kernel(abs_r, max_radius_h):
    h = max_radius_h / 2
    q = abs_r / h
    factor_alpha = (7 / (4 * np.pi * h**2)) / h
    if 0.0 <= q <= 2.0:
        return factor_alpha * -5 * q(1 - q / 2) ** 3
    elif 2 < q:
        return 0
    else:
        raise Exception("Wendland_kernel smth is wrong here")


def artifical_viscosity_term(particle_a, particle_b, alpha, beta, velo_ab, r_ab):
    pi_ab = 0
    if (velo_ab).dot(
        r_ab
    ) < 0:  # Not sure if particle_a and particle_b must be switched
        mean_c = (particle_a.c_sound + particle_b.c_sound) / 2
        mean_rho = (particle_a.rho + particle_b.rho) / 2
        mean_h = (particle_a.radius_of_kernel + particle_b.radius_of_kernel) / 2
        nu_ab = (mean_h * velo_ab.dot(r_ab)) / (r_ab.dot(r_ab) + 0.001**2)
        pi_ab = (-alpha * mean_c * nu_ab + beta * nu_ab**2) / mean_rho
    return pi_ab


def nearest_neighbor_sph_force(A, particle_a, prio_q, degree_of_freedom):
    force_a = particle_a.c_sound**2 / (degree_of_freedom * particle_a.rho)

    particle_a.radius_of_kernel = prio_q.key()

    energy_dt = 0.0
    acc = 0.0  # is this zero?
    for q in prio_q.heap:
        particle_b = A[q[1]]
        max_dist = prio_q.key()
        radius = np.sqrt(-q[0])
        force_b = particle_b.c_sound**2 / (degree_of_freedom / particle_b.rho)
        r = r_diff(particle_a, particle_b)
        abs_r = np.sqrt(r.dot(r))
        # print("RAD", radius, abs_r)

        energy_dt += particle_b.mass * (particle_a.velo - particle_b.velo).dot(
            gradient_monaghan_kernel(r / abs_r, radius, max_dist)
        )
        # print(energy_dt, " += ",   particle_b.mass, " * ", (particle_a.velo - particle_b.velo), " DOT ", gradient_monaghan_kernel(r/abs_r, radius, max_dist), " = ", (particle_a.velo - particle_b.velo).dot(gradient_monaghan_kernel(r/abs_r, radius, max_dist)))

        acc -= (
            particle_b.mass
            * (
                force_a
                + force_b
                + artifical_viscosity_term(
                    particle_a, particle_b, 1, 2, particle_a.velo - particle_b.velo, r
                )
            )
            * gradient_monaghan_kernel(r / abs_r, radius, max_dist)
        )
        # acc -= particle_b.mass * (force_a + force_b) * gradient_monaghan_kernel(particle_a, particle_b, prio_q.key())
        # print("MASS: ", particle_b.mass , "FA: ", force_a, "FB",force_b, "viscosity", artifical_viscosity_term(particle_a, particle_b, 1, 2, particle_a.velo - particle_b.velo, r_diff(particle_a, particle_b)), "gradient_monaghan_kernel: ", gradient_monaghan_kernel(particle_a, particle_b, prio_q.key()))
        # print("ALLINONE: " ,particle_b.mass * (force_a + force_b + artifical_viscosity_term(particle_a, particle_b, 1, 2, particle_a.velo - particle_b.velo, r_diff(particle_a, particle_b))) * gradient_monaghan_kernel(particle_a, particle_b, particle_a.radius_of_kernel))

    energy_dt *= force_a

    return acc, energy_dt


def calculate_sound(particle_a, degree_of_freedom):
    return np.sqrt(
        particle_a.predicted_energy * degree_of_freedom * (degree_of_freedom - 1)
    )


def calculate_forces(method, A):
    root = cell([0, 0], [1, 1], 0, len(A) - 1, 0)
    treebuild(A, root, 0)
    for q in A:
        q.prio_q = prioqueue(q.k_nearest)
        neighbor_search_periodic(q.prio_q, root, A, q.r, np.array([1, 1]))
        q.rho = density(method, A, q.prio_q)
        q.c_sound = calculate_sound(q, 2.0)
    for q in A:
        q.acc, q.energy_dt = nearest_neighbor_sph_force(A, q, q.prio_q, 2.0)


def initial_setup(method, A):
    # print("INIT BFD: ",A[0])
    first_drift(A, 0)
    # print("INIT AFD BCF: ",A[0])
    calculate_forces(method, A)
    # print("INIT ACF: ",A[0])


def smooth_particle_hydrodynamics(method, A, time_step):
    # print("BFD: ",A[0])
    first_drift(A, time_step / 2)
    # print("AFD BCF: ",A[0])
    calculate_forces(method, A)
    # print("ACF BK: ",A[0])
    kick(A, time_step)
    # print("AK BSD: ",A[0])
    second_drift(A, time_step / 2)
    # print("ASD: ",A[0])


if __name__ == "__main__":
    nr_particles = 256
    k_nearest = 32
    A = random_AMatrix(nr_particles, k_nearest)
    fig, axs = plt.subplots()

    initial_setup(monaghan_kernel, A)

    x = [p.r[0] for p in A]
    y = [p.r[1] for p in A]
    scatter = axs.scatter(x, y, c="r", marker="o")
    time_step = 0

    def update(frame):
        global time_step
        time_step += 0.001
        smooth_particle_hydrodynamics(monaghan_kernel, A, time_step)
        scatter.set_offsets([p.r for p in A])
        print("frame: ", frame)

    ani = FuncAnimation(fig, update, frames=range(50), interval=10, repeat=False)
    ani.save("sph_500.mp4")

    # plt.show()
