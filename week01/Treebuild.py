# -*- coding: utf-8 -*-
"""week01.ipynb

@author: Dennys
"""

import numpy as np
import random as rd
import matplotlib.pyplot as plt

class particle:
    def __init__(self, r):
        self.r = r  # position of the particle [x,y]
        self.rho = 0.0  # density of the particle
        # ...  more properties of the particle

    def __repr__(self):
        return "(x: %s, y: %s)" % (self.r[0], self.r[1])

class cell:
    def __init__(self, rLow, rHigh, lower, upper):
        self.rLow = rLow  # [xMin, yMin]
        self.rHigh = rHigh  # [xMax, yMax]
        self.iLower = lower  # index to first particle in particle array
        self.iUpper = upper  # index to last particle in particle array
        self.pLower = None  # reference to tree cell for lower part
        self.pUpper = None  # reference to tree cell for upper part

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
            start += 1; end -= 1;
        elif subA[start].r[d] >= v and subA[end].r[d] >= v:
            end -= 1
        elif subA[start].r[d] < v and subA[end].r[d] < v:
            start += 1
        elif subA[start].r[d] >= v and subA[end].r[d] < v:
            subA[start], subA[end] = subA[end], subA[start]
            start += 1; end -= 1
    return start

def treebuild(A, root, dim):
   
    v = 0.5 * (root.rLow[dim] + root.rHigh[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)

    if s != 0:
        rHigh = root.rHigh[:]
        rHigh[dim] = v
        cLow = cell(root.rLow, rHigh, root.iLower, s-1)
        root.pLower = cLow;
        if len(A[:s]) > 8: 
            treebuild(A[:s], cLow, 1-dim)

    if s <= len(A):
        rLow = root.rLow[:]
        rLow[dim] = v
        cHigh = cell(rLow, root.rHigh, 0, root.iUpper-s)
        root.pUpper = cHigh
        if len(A[s:]) > 8: 
            treebuild(A[s:], cHigh, 1-dim)

def random_AMatrix(nr_particles):
    P = []
    A = np.array([])
    for _ in range(nr_particles):
        p = [rd.random(), rd.random()]
        P.append(p)

    P = np.asarray(P)
    P = P[P[:,0].argsort()]

    for p in P:
        a = np.array([particle([p[0], p[1]])])
        A = np.append(A, a)
    return A

def plottree(root):
    #draw a rectangle specified by rLow and rHigh
    if root.pLower:
        plottree(root.pLower)
    if root.pUpper:
        plottree(root.pUpper)
    xl, xh = root.rLow[0], root.rHigh[0]
    yl, yh = root.rLow[1], root.rHigh[1]
    plt.plot([xl,xh],[yl,yl], color='red')
    plt.plot([xl,xh],[yh,yh], color='red')
    plt.plot([xl,xl],[yl,yh], color='blue')
    plt.plot([xh,xh],[yl,yh], color='blue')

if __name__ == "__main__":
    A = random_AMatrix(100)
    root_rlow = [0,0]
    root_rhigh = [1,1]

    root = cell(root_rlow, root_rhigh, 0, len(A)-1)

    treebuild(A, root, 0)

    for p in A:
      plt.scatter(p.r[0],p.r[1], color = "k")
    plt.xlim(root_rlow[0], root_rhigh[0])
    plt.ylim(root_rlow[1], root_rhigh[1])
    plottree(root)

    plt.show()
