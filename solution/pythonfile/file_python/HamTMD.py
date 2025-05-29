import typing
from numpy import dtype, sqrt, exp, pi, zeros

import numpy as np


from numpy import typing as nt


def Hamiltonian(band, argument, alattice, p, q, pbc, kx, ky, E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6):
    # matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(argument)
    eta = p / (1 * q)

    alpha = 1 / 2 * kx * alattice
    beta = sqrt(3) / 2 * ky * alattice

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

    for m in range(0, q):
        h0[m][m] = E_R0[0][0]
        h1[m][m] = E_R0[0][1]
        h2[m][m] = E_R0[0][2]
        h1T[m][m] = E_R0[1][0]
        h11[m][m] = E_R0[1][1]
        h12[m][m] = E_R0[1][2]
        h2T[m][m] = E_R0[2][0]
        h12T[m][m] = E_R0[2][1]
        h22[m][m] = E_R0[2][2]

        phaseR1 = exp(2j * alpha)
        phaseR4 = exp(-2j * alpha)
        phaseR2 = exp(1j * (alpha - beta))
        phaseR3 = exp(1j * (-alpha - beta))
        phaseR5 = exp(1j * (-alpha + beta))
        phaseR6 = exp(1j * (alpha + beta))

        h0[m, pbc(m + 2, q)] = E_R1[0][0] * phaseR1
        h1[m, pbc(m + 2, q)] = E_R1[0][1] * phaseR1
        h2[m, pbc(m + 2, q)] = E_R1[0][2] * phaseR1
        h1T[m, pbc(m + 2, q)] = E_R1[1][0] * np.conjugate(phaseR1)
        h11[m, pbc(m + 2, q)] = E_R1[1][1] * phaseR1
        h12[m, pbc(m + 2, q)] = E_R1[1][2] * phaseR1
        h2T[m, pbc(m + 2, q)] = E_R1[2][0] * np.conjugate(phaseR1)
        h12T[m, pbc(m + 2, q)] = E_R1[2][1] * np.conjugate(phaseR1)
        h22[m, pbc(m + 2, q)] = E_R1[2][2] * phaseR1

        h0[m, pbc(m + 1, q)] = E_R2[0][0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[0][0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h1[m, pbc(m + 1, q)] = E_R2[0][1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[0][1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h2[m, pbc(m + 1, q)] = E_R2[0][2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[0][2] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h1T[m, pbc(m + 1, q)] = E_R2[1][0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * np.conjugate(phaseR2) + E_R6[1][0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * np.conjugate(phaseR6)
        h11[m, pbc(m + 1, q)] = E_R2[1][1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[1][1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h12[m, pbc(m + 1, q)] = E_R2[1][2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[1][2] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6
        h2T[m, pbc(m + 1, q)] = E_R2[2][0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * np.conjugate(phaseR2) + E_R6[2][0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * np.conjugate(phaseR6)
        h12T[m, pbc(m + 1, q)] = E_R2[2][1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * np.conjugate(phaseR2) + E_R6[2][1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * np.conjugate(phaseR6)
        h22[m, pbc(m + 1, q)] = E_R2[2][2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) * phaseR2 + E_R6[2][2] * exp(1j * 2 * pi * (m + 1 / 2) * eta) * phaseR6

        h0[m, pbc(m - 1, q)] = E_R5[0][0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[0][0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h1[m, pbc(m - 1, q)] = E_R5[0][1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[0][1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h2[m, pbc(m - 1, q)] = E_R5[0][2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[0][2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h1T[m, pbc(m - 1, q)] = E_R5[1][0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * np.conjugate(phaseR5) + E_R3[1][0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * np.conjugate(phaseR3)
        h11[m, pbc(m - 1, q)] = E_R5[1][1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[1][1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h12[m, pbc(m - 1, q)] = E_R5[1][2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * phaseR5 + E_R3[1][2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3
        h2T[m, pbc(m - 1, q)] = E_R5[2][0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * np.conjugate(phaseR5) + E_R3[2][0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * np.conjugate(phaseR3)
        h12T[m, pbc(m - 1, q)] = E_R5[2][1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * np.conjugate(phaseR5) + E_R3[2][1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) * np.conjugate(phaseR3)
        h22[m, pbc(m - 1, q)] = E_R5[2][2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + E_R3[2][2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) * phaseR3

        h0[m, pbc(m - 2, q)] = E_R4[0][0] * phaseR4
        h1[m, pbc(m - 2, q)] = E_R4[0][1] * phaseR4
        h2[m, pbc(m - 2, q)] = E_R4[0][2] * phaseR4
        h1T[m, pbc(m - 2, q)] = E_R4[1][0] * np.conjugate(phaseR4)
        h11[m, pbc(m - 2, q)] = E_R4[1][1] * phaseR4
        h12[m, pbc(m - 2, q)] = E_R4[1][2] * phaseR4
        h2T[m, pbc(m - 2, q)] = E_R4[2][0] * np.conjugate(phaseR4)
        h12T[m, pbc(m - 2, q)] = E_R4[2][1] * np.conjugate(phaseR4)
        h22[m, pbc(m - 2, q)] = E_R4[2][2] * phaseR4

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
