# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:04:21 2023

@author: Stefan
"""

import numpy as np

file = "ch130.tsp"
x, y = np.loadtxt(file, delimiter=' ', comments="EOF",
                  skiprows=6, usecols=(1, 2), unpack=True)

print("# data read: %i" % (len(x)))
for i in range(len(x)):
    xi = x[i]
    yi = y[i]
    print("(x_%i, y_%i) = (%f, %f)" % (i, i, xi, yi))

file = "ch130.opt.tour"
opttour = np.loadtxt(file, delimiter=' ', comments="-1",
                  dtype=int, skiprows=5, usecols=(0), unpack=True)
print("# data read: %i" % (len(opttour)))
print("Optimal tour\n", opttour)
