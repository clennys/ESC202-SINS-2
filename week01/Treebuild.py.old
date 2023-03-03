# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:41:47 2021

@author: Dennys
"""

import numpy as np
import random as rd

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

def printsubA(A, i, j):
    subA = A[i : j + 1]
    for e in subA:
        print("submat", e)

def findXYMax(A):
    length = 0.0
    max = A[0]
    for p in A:
        tmp = np.linalg.norm(p.r)
        if length < tmp:
            length = tmp
            max = p
    return max

def findXYMin(A):
    length = 0.0
    min = A[0]
    for p in A:
        tmp = np.linalg.norm(p.r)
        if length > tmp:
            length = tmp
            min = p
    return min
    

def treebuild(A, root, dim):
   
    v = 0.5 * (root.rLow[dim] + root.rHigh[dim])
    s = partition(A, root.iLower, root.iUpper, v, dim)
    dim = 1 if dim == 0 else 0;

    if s > 0 and len(A[root.iLower : s-1]) > 0:
        max = findXYMax(A[root.iLower: s - 1])
        min = findXYMin(A[root.iLower: s - 1])
        cLow = cell(min.r, max.r, root.iLower, s-1)
        root.pLower = cLow;
        if len(A[root.iLower: s-1]) + 1 > 8: 
            treebuild(A, cLow, dim)

    s += root.iLower
    if len(A[s : root.iUpper]) > 0:
        max = findXYMax(A[s: root.iUpper])
        min = findXYMin(A[s: root.iUpper])
        cHigh = cell(min.r, max.r, s, root.iUpper)
        root.pUpper = cHigh
        if len(A[s: root.iUpper])+1 > 8: 
            treebuild(A, cHigh, dim)
    
    # may have two parts: lower..s-1 and s..upper
    # if there is a lower part:
    #     cLow = cell(..., root.iLower, s-1)
    #     root.pLower = cLow
    #     if there are more than 8 particles in cell:
    #         treebuild(A, cLow, 1 - dim)
    # if there is an upper part:
    #     cHigh = cell(..., s, root.iUpper)
    #     root.pUpper = cHigh
    #     if there are more than 8 particles in cell:
    #         treebuild(A, cHigh, 1 - dim)
    # grafical representation of tree

    
def plottree(root):
    print("[%s, %s]" % (root.iLower, root.iUpper)) 
    if root.pLower is not None:
        print("Lower Child: ")
        plottree(root.pLower)
    if root.pUpper is not None:
        print("Upper Child: ")
        plottree(root.pUpper)

   
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



if __name__ == "__main__":

    A = random_AMatrix(20)

    root = cell(A[0].r, A[-1].r, 0, len(A)-1)


    treebuild(A, root, 0)


    plottree(root)


# r = np.array([0, 1])
# p = particle(r)
# print(p.r)
#
# # Create array A with particles
#
# # Test partition function
# def test1():
#     A = [] # initialize with particles with sequential coordinates in x, same yMax
#     s = partition(A,0,10,0.5,0)
#     if s==5:
#         return True
#     return False
#
# # def test2():
#
# # def testAll():
# #     if !test1():
# #         return False
# #     ...
# #     return True
#     
# # Build the tree
#
# rLow = np.array([0, 0])
# rHigh = np.array([1, 1])
# lower = 0
# upper = last index of A
# root = cell(rLow, rHigh, lower, upper)
# dim = 0
# treebuild(A, root, dim)
#
# plottree(root)
#
#
# # O(N**2) Test Code
# # k = Number of nearest neighbors
# for p in A:
#     NN = []
#     d2NN = []
#     for i in range(k):
#         d2min = float('inf')
#         for q in A:
#             if p != q and q not in NN:
#                 d2 = p.dist2(q)
#                 if d2 < d2min:
#                     d2min = d2
#                     qmin = q
#         NN.append(qmin)
#         d2NN.append(d2min)
#     # Here NN and d2NN lists for particle p are filled.
#     # Compare them with the lists you got from the recursive algorithm
#         
#
# # Priority Queue
# # https://docs.python.org/3/library/heapq.html
# # Use a tuple (key, data)
#
# # Write a wrapper class that implements our interface (replace() and dist2())
# # but uses heapq for the implementation.
#
# from heapq import *
# import numpy as np
#
# heap = []
# sentinel = (-np.inf, None)
# heappush(heap, sentinel)
# heappush(heap, sentinel)
# heappush(heap, sentinel)
# print(heap)
# heapreplace(heap, (-2,'p2'))
# print(heap)
# heapreplace(heap, (-5,'p5'))
# print(heap)
# heapreplace(heap, (-4,'p4'))
# print(heap)
# maxdist = -heap[0][0]
# print(maxdist)
#
#
# heap = []
# sentinel = (0, None)
# heappush(heap, sentinel)
# heappush(heap, sentinel)
# heappush(heap, sentinel)
# print(heap)
# heapreplace(heap, (1.0/2,'p2'))
# print(heap)
# heapreplace(heap, (1.0/5,'p5'))
# print(heap)
# heapreplace(heap, (1.0/4,'p4'))
# print(heap)
# maxdist = 1.0/heap[0][0]
# print(maxdist)
#
