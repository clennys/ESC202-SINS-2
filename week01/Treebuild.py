# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:41:47 2021

@author: Stefan
"""

import numpy as np

class particle:
    def __init__(self, r):
        self.r = r  # position of the particle [x,y]
        self.rho = 0.0  # density of the particle
        # ...  more properties of the particle

class cell:
    def __init__(self, rLow, rHigh, lower, upper):
        self.rLow = rLow  # [xMin, yMin]
        self.rHigh = rHigh  # [yMax, yMax]
        self.iLower = lower  # index to first particle in particle array
        self.iUpper = upper  # index to last particle in particle array
        self.pLower = None  # reference to tree cell for lower part
        self.pUpper = None  # reference to tree cell for upper part

def partition(A, i, j, v, d):
    """
    Input:
      A: array of all particles
      i: start index of subarray for partition
      j: end index (inclusive) for subarray for partition
      v: value for partition e.g. 0.5
      d: dimension to use for partition, 0 (for x) or 1 (for y)
    """
    # ...
    # return s


def treebuild(A, root, dim):
   
    v = 0.5 * (root.rLow[dim] + root.rHigh[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)
    
    # may have two parts: lower..s-1 and s..upper
    if there is a lower part:
        cLow = cell(..., root.iLower, s-1)
        root.pLower = cLow
        if there are more than 8 particles in cell:
            treebuild(A, cLow, 1 - dim)
    if there is an upper part:
        cHigh = cell(..., s, root.iUpper)
        root.pUpper = cHigh
        if there are more than 8 particles in cell:
            treebuild(A, cHigh, 1 - dim)
    # grafical representation of tree
    
def plottree(root):
    draw a rectangle specified by rLow and rHigh
    
r = np.array([0, 1])
p = particle(r)
print(p.r)

# Create array A with particles

# Test partition function
def test1():
    A = initialize with particles with sequential coordinates in x, same yMax
    s = partition(A,0,10,0.5,0)
    if s=5:
        return True
    return False

def test2():

def testAll():
    if !test1():
        return False
    ...
    return True
    
# Build the tree

rLow = np.array([0, 0])
rHigh = np.array([1, 1])
lower = 0
upper = last index of A
root = cell(rLow, rHigh, lower, upper)
dim = 0
treebuild(A, root, dim)

plottree(root)


# O(N**2) Test Code
# k = Number of nearest neighbors
for p in A:
    NN = []
    d2NN = []
    for i in range(k):
        d2min = float('inf')
        for q in A:
            if p != q and q not in NN:
                d2 = p.dist2(q)
                if d2 < d2min:
                    d2min = d2
                    qmin = q
        NN.append(qmin)
        d2NN.append(d2min)
    # Here NN and d2NN lists for particle p are filled.
    # Compare them with the lists you got from the recursive algorithm
        

# Priority Queue
# https://docs.python.org/3/library/heapq.html
# Use a tuple (key, data)

# Write a wrapper class that implements our interface (replace() and dist2())
# but uses heapq for the implementation.

from heapq import *
import numpy as np

heap = []
sentinel = (-np.inf, None)
heappush(heap, sentinel)
heappush(heap, sentinel)
heappush(heap, sentinel)
print(heap)
heapreplace(heap, (-2,'p2'))
print(heap)
heapreplace(heap, (-5,'p5'))
print(heap)
heapreplace(heap, (-4,'p4'))
print(heap)
maxdist = -heap[0][0]
print(maxdist)


from heapq import *

heap = []
sentinel = (0, None)
heappush(heap, sentinel)
heappush(heap, sentinel)
heappush(heap, sentinel)
print(heap)
heapreplace(heap, (1.0/2,'p2'))
print(heap)
heapreplace(heap, (1.0/5,'p5'))
print(heap)
heapreplace(heap, (1.0/4,'p4'))
print(heap)
maxdist = 1.0/heap[0][0]
print(maxdist)

