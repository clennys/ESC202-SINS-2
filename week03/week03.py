# -*- coding: utf-8 -*-
"""week03.ipynb

@author Dennys Huber
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import heapq as hq


class particle:
    def __init__(self, r, m=1):
        self.r = r  # position of the particle [x,y]
        self.rho = 0.0  # density of the particle
        self.mass = m
        # ...  more properties of the particle

    def __repr__(self):
        return "(x: %s, y: %s, rho: %s, m: %s)" % (
            self.r[0],
            self.r[1],
            self.rho,
            self.mass,
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


def random_AMatrix(nr_particles):
    P = []
    A = np.array([])
    for _ in range(nr_particles):
        p = [rd.random(), rd.random()]
        P.append(p)

    P = np.asarray(P)
    P = P[P[:, 0].argsort()]

    for p in P:
        a = np.array([particle([p[0], p[1]])])
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


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 2)

    nr_particles = 1000
    k_nearest = 32
    A = random_AMatrix(nr_particles)
    root_rlow = [0, 0]
    root_rhigh = [1, 1]

    root = cell(root_rlow, root_rhigh, 0, len(A) - 1, 0)

    treebuild(A, root, 0)

    for i in range(2):
        ax[i].set_xlim(root_rlow[0], root_rhigh[0])
        ax[i].set_ylim(root_rlow[1], root_rhigh[1])

    densities_top, x_top, y_top = plot_density(top_hat_kernel, k_nearest, A, root)
    densities_mon, x_mon, y_mon = plot_density(monaghan_kernel, k_nearest, A, root)

    sp0 = ax[0].scatter(x_top, y_top, c=densities_top, cmap="coolwarm")
    sp1 = ax[1].scatter(x_mon, y_mon, c=densities_mon, cmap="coolwarm")
    plt.colorbar(sp0, ax=ax[0])
    plt.colorbar(sp1, ax=ax[1])

    ax[0].set_title(
        "Top Hat Kernel with %s nearest neighbors and %s particles"
        % (k_nearest, nr_particles)
    )
    ax[1].set_title(
        "Monaghan Kernel with %s nearest neighbors and %s particles"
        % (k_nearest, nr_particles)
    )

    plt.show()
