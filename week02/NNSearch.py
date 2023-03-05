
def neighbor_search_periodic(pq, root, particles, r, period):
    # walk the closest image first (at offset=[0, 0])
    for y in [0.0, -period[1], period[1]]:
        for x in [0.0, -period[0], period[0]]:
            rOffset = np.array([x, y])
            neighbor_search(pq, root, particles, r, rOffset)


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
        for j in range(root.iLower, root.iUpper):
            d2 = dist2(particles[j].r, ri)
            if d2 < pq.key():
                pq.replace(d2, j, particles[j].r - rOffset)
    # for pq write a wrapper class that implements key() and replace() using heapq package

# Add these to the Cell class

def celldist2(self, r):
    """Calculates the squared minimum distance between a particle
    position and this node."""
    d1 = r - self.rHigh
    d2 = self.rLow - r
    d1 = np.maximum(d1, d2)
    d1 = np.maximum(d1, np.zeros_like(d1))
    return d1.dot(d1)