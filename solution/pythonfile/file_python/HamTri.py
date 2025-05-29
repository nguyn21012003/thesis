import numpy as np

from numpy import sqrt, pi, sin, cos, exp


def HamTriangular(band, argument, alattice, p, q, pbc, kx, ky, E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6):

    eta = p / q

    alpha = 1 / 2 * kx * alattice * 0
    beta = sqrt(3) / 2 * ky * alattice * 0

    h0 = np.zeros([q, q], dtype=complex)
    h1 = np.zeros([q, q], dtype=complex)
    h1T = np.zeros([q, q], dtype=complex)
    h2 = np.zeros([q, q], dtype=complex)
    h2T = np.zeros([q, q], dtype=complex)
    h11 = np.zeros([q, q], dtype=complex)
    h22 = np.zeros([q, q], dtype=complex)
    h12 = np.zeros([q, q], dtype=complex)
    h12T = np.zeros([q, q], dtype=complex)
    H2band = np.zeros([2 * q, 2 * q], dtype=complex)
    H = np.zeros([3 * q, 3 * q], dtype=complex)

    for m in range(q):
        phaseR1 = exp(2j * alpha)
        phaseR4 = exp(-2j * alpha)
        phaseR2 = exp(2j * (alpha - beta))
        phaseR3 = exp(-2j * (alpha + beta))
        phaseR5 = exp(-2j * (alpha - beta))
        phaseR6 = exp(2j * (alpha + beta))

        h0[m, m] = E_R0[0][0] + phaseR3 * exp(-2j * pi * (m + 1 / 4) * eta) * E_R3[0][0] + phaseR6 * exp(2j * pi * (m + 1 / 4) * eta) * E_R6[0][0]
        h0[m, pbc(m + 1, q)] = phaseR1 * E_R1[0][0] + phaseR2 * E_R2[0][0] * exp(-2j * pi * (m - 1 / 4) * eta)
        h0[m, pbc(m - 1, q)] = phaseR4 * E_R4[0][0] + phaseR5 * E_R5[0][0] * exp(2j * pi * (m - 1 / 4) * eta)

        h1[m, m] = E_R0[0][1] + phaseR3 * exp(-2j * pi * (m + 1 / 4) * eta) * E_R3[0][1] + phaseR6 * exp(2j * pi * (m + 1 / 4) * eta) * E_R6[0][1]
        h1[m, pbc(m + 1, q)] = phaseR1 * E_R1[0][1] + phaseR2 * E_R2[0][1] * exp(-2j * pi * (m - 1 / 4) * eta)
        h1[m, pbc(m - 1, q)] = phaseR4 * E_R4[0][1] + phaseR5 * E_R5[0][1] * exp(2j * pi * (m - 1 / 4) * eta)

        h2[m, m] = E_R0[0][2] + phaseR3 * exp(-2j * pi * (m + 1 / 4) * eta) * E_R3[0][2] + phaseR6 * exp(2j * pi * (m + 1 / 4) * eta) * E_R6[0][2]
        h2[m, pbc(m + 1, q)] = phaseR1 * E_R1[0][2] + phaseR2 * E_R2[0][2] * exp(-2j * pi * (m - 1 / 4) * eta)
        h2[m, pbc(m - 1, q)] = phaseR4 * E_R4[0][2] + phaseR5 * E_R5[0][2] * exp(2j * pi * (m - 1 / 4) * eta)

        h1T[m, m] = E_R0[1][0] + np.conjugate(phaseR3) * exp(2j * pi * (m + 1 / 4) * eta) * E_R3[1][0] + np.conjugate(phaseR6) * exp(-2j * pi * (m + 1 / 4) * eta) * E_R6[1][0]
        h1T[m, pbc(m + 1, q)] = np.conjugate(phaseR1) * E_R1[1][0] + np.conjugate(phaseR2) * E_R2[1][0] * exp(2j * pi * (m - 1 / 4) * eta)
        h1T[m, pbc(m - 1, q)] = np.conjugate(phaseR4) * E_R4[1][0] + np.conjugate(phaseR5) * E_R5[1][0] * exp(-2j * pi * (m - 1 / 4) * eta)

        h11[m, m] = E_R0[1][1] + phaseR3 * exp(-2j * pi * (m + 1 / 4) * eta) * E_R3[1][1] + phaseR6 * exp(2j * pi * (m + 1 / 4) * eta) * E_R6[1][1]
        h11[m, pbc(m + 1, q)] = phaseR1 * E_R1[1][1] + phaseR2 * E_R2[1][1] * exp(-2j * pi * (m - 1 / 4) * eta)
        h11[m, pbc(m - 1, q)] = phaseR4 * E_R4[1][1] + phaseR5 * E_R5[1][1] * exp(2j * pi * (m - 1 / 4) * eta)

        h12[m, m] = E_R0[1][2] + phaseR3 * exp(-2j * pi * (m + 1 / 4) * eta) * E_R3[1][2] + phaseR6 * exp(2j * pi * (m + 1 / 4) * eta) * E_R6[1][2]
        h12[m, pbc(m + 1, q)] = phaseR1 * E_R1[1][0] + phaseR2 * E_R2[1][2] * exp(-2j * pi * (m - 1 / 4) * eta)
        h12[m, pbc(m - 1, q)] = phaseR4 * E_R4[1][0] + phaseR5 * E_R5[1][2] * exp(2j * pi * (m - 1 / 4) * eta)

        h2T[m, m] = E_R0[2][0] + np.conjugate(phaseR3) * exp(2j * pi * (m + 1 / 4) * eta) * E_R3[2][0] + np.conjugate(phaseR6) * exp(-2j * pi * (m + 1 / 4) * eta) * E_R6[2][0]
        h2T[m, pbc(m + 1, q)] = np.conjugate(phaseR1) * E_R1[2][0] + np.conjugate(phaseR2) * E_R2[2][0] * exp(2j * pi * (m - 1 / 4) * eta)
        h2T[m, pbc(m - 1, q)] = np.conjugate(phaseR4) * E_R4[2][0] + np.conjugate(phaseR5) * E_R5[2][0] * exp(-2j * pi * (m - 1 / 4) * eta)

        h12T[m, m] = E_R0[2][1] + np.conjugate(phaseR3) * exp(2j * pi * (m + 1 / 4) * eta) * E_R3[2][1] + np.conjugate(phaseR6) * exp(-2j * pi * (m + 1 / 4) * eta) * E_R6[2][1]
        h12T[m, pbc(m + 1, q)] = np.conjugate(phaseR1) * E_R1[2][1] + np.conjugate(phaseR2) * E_R2[2][1] * exp(2j * pi * (m - 1 / 4) * eta)
        h12T[m, pbc(m - 1, q)] = np.conjugate(phaseR4) * E_R4[2][1] + np.conjugate(phaseR5) * E_R5[2][1] * exp(-2j * pi * (m - 1 / 4) * eta)

        h22[m, m] = E_R0[2][2] + phaseR3 * exp(-2j * pi * (m + 1 / 4) * eta) * E_R3[2][2] + phaseR6 * exp(2j * pi * (m + 1 / 4) * eta) * E_R6[2][2]
        h22[m, pbc(m + 1, q)] = phaseR1 * E_R1[2][2] + phaseR2 * E_R2[2][2] * exp(-2j * pi * (m - 1 / 4) * eta)
        h22[m, pbc(m - 1, q)] = phaseR4 * E_R4[2][2] + phaseR5 * E_R5[2][2] * exp(2j * pi * (m - 1 / 4) * eta)

    if band == 1:
        return h0

    elif band == 2:
        H2band[0:q, 0:q] = h1
        H2band[0:q, q : 2 * q] = h2
        H2band[q : 2 * q, 0:q] = h1T
        H2band[q : 2 * q, q : 2 * q] = h2T
        return H2band

    elif band == 3:
        H[0:q, 0:q] = h0
        H[0:q, q : 2 * q] = h1
        H[0:q, 2 * q : 3 * q] = h2
        H[q : 2 * q, 0:q] = h1T
        H[q : 2 * q, q : 2 * q] = h11
        H[q : 2 * q, 2 * q : 3 * q] = h12
        H[2 * q : 3 * q, 0:q] = h2T
        H[2 * q : 3 * q, q : 2 * q] = h12T
        H[2 * q : 3 * q, 2 * q : 3 * q] = h22

        return H
