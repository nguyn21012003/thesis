import numpy as np

from numpy import sqrt, pi, sin, cos, exp


def HamTriNN(band, argument, alattice, p, q, pbc, kx, ky):

    eta = p / q

    alpha = 1 / 2 * kx * alattice * 0
    beta = sqrt(3) / 2 * ky * alattice * 0

    h0 = np.zeros([q, q], dtype=complex)

    t = 1
    tt = 0
    ta = 1
    tb = 1
    tc = 1
    taa = 1
    tbb = 1
    tcc = 1
    for m in range(q):

        h0[m, m] = 2 * tt * cos(2 * pi * eta * m)
        h0[m, pbc(m + 2, q)] = t
        h0[m, pbc(m - 2, q)] = tt
        h0[m, pbc(m + 1, q)] = 2 * t * cos(pi * eta * (m - 1 / 2))
        h0[m, pbc(m - 1, q)] = 2 * t * cos(pi * eta * (m - 1 - 1 / 2))
        h0[m, pbc(m + 3, q)] = 2 * tt * cos(pi * eta * (m + 1 - 1 / 2))
        h0[m, pbc(m - 3, q)] = 2 * tt * cos(pi * eta * (m - 2 - 1 / 2))

    return h0
