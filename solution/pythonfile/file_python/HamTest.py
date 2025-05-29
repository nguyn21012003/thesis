import numpy as np
from numpy import exp, pi, sqrt


def HamTNN_test(band, argument, alattice, p, q, pbc, kx, ky, E0, h1, h2, h3, h4, h5, h6, o1, o2, o3, o4, o5, o6, v1, v2, v3, v4, v5, v6):
    # matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(argument)
    eta = p / q
    alpha = 1 / 2 * kx * alattice
    beta = sqrt(3) / 2 * ky * alattice

    H0 = np.zeros([q, q], dtype=complex)
    H1 = np.zeros([q, q], dtype=complex)
    H1T = np.zeros([q, q], dtype=complex)
    H2 = np.zeros([q, q], dtype=complex)
    H2T = np.zeros([q, q], dtype=complex)
    H11 = np.zeros([q, q], dtype=complex)
    H22 = np.zeros([q, q], dtype=complex)
    H12 = np.zeros([q, q], dtype=complex)
    H12T = np.zeros([q, q], dtype=complex)
    H = np.zeros([3 * q, 3 * q], dtype=complex)

    for m in range(0, q):

        H0[m, m] = E0[0, 0] + v2[0, 0] * exp(-4j * pi * m * eta) + v5[0, 0] * exp(4j * pi * m * eta)

        H0[m, pbc(m + 3, q)] = v1[0, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta)

        H0[m, pbc(m - 3, q)] = v3[0, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta)

        H0[m, pbc(m + 2, q)] = h1[0, 0] + o2[0, 0] * exp(-4j * pi * (m + 1) * eta) + o6[0, 0] * exp(4j * pi * (m + 1) * eta)

        H0[m, pbc(m + 4, q)] = o1[0, 0]

        H0[m, pbc(m + 1, q)] = h2[0, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta)

        H0[m, pbc(m - 1, q)] = h5[0, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)

        H0[m, pbc(m - 2, q)] = h4[0, 0] + o3[0, 0] * exp(-4j * pi * (m - 1) * eta) + o5[0, 0] * exp(4j * pi * (m - 1) * eta)

        H0[m, pbc(m - 4, q)] = o4[0, 0]

    if band == 1:
        return H0
