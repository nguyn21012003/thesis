import numpy as np
from numpy import exp, pi, sqrt
from numpy.typing import NDArray

from file_python.condition import pbc


def HamTNN(band: int, alattice: float, p: int, q: int, kx: float, ky: float, IM: dict) -> NDArray:
    """[Summary]

    Define a Hamiltonian for the TMD 3 band using Ref Phys.rev.B 88,085433

    Args:
        band: is the number of the band considering. Default is 3, you can choose 1.
        alattice: is the lattice constant.
        p: numerator.
        q: denomitor in the magnetic
        kx: k valley
        ky: k valley
        IM: hopping terms using group theory and Irreducible matrices.
    """
    E0 = IM["NN"][0]
    h1 = IM["NN"][1]
    h2 = IM["NN"][2]
    h3 = IM["NN"][3]
    h4 = IM["NN"][4]
    h5 = IM["NN"][5]
    h6 = IM["NN"][6]
    o1 = IM["TNN"][0]
    o2 = IM["TNN"][1]
    o3 = IM["TNN"][2]
    o4 = IM["TNN"][3]
    o5 = IM["TNN"][4]
    o6 = IM["TNN"][5]
    v1 = IM["NNN"][0]
    v2 = IM["NNN"][1]
    v3 = IM["NNN"][2]
    v4 = IM["NNN"][3]
    v5 = IM["NNN"][4]
    v6 = IM["NNN"][5]

    eta = 1 * p / q
    alpha = 1 / 2 * kx * alattice
    beta = sqrt(3) / 2 * ky * alattice

    lambd = 0

    hR1 = exp(1j * 2 * alpha)
    hR2 = exp(1j * (alpha - beta))
    hR3 = exp(1j * (-alpha - beta))
    hR4 = exp(-1j * 2 * alpha)
    hR5 = exp(1j * (-alpha + beta))
    hR6 = exp(1j * (alpha + beta))

    vR1 = exp(1j * (3 * alpha - beta))
    vR2 = exp(1j * (-2 * beta))
    vR3 = exp(1j * (-3 * alpha - beta))
    vR4 = exp(1j * (-3 * alpha + beta))
    vR5 = exp(1j * (2 * beta))
    vR6 = exp(1j * (3 * alpha + beta))

    oR1 = exp(1j * 4 * alpha)
    oR2 = exp(2j * (alpha - beta))
    oR3 = exp(2j * (-alpha - beta))
    oR4 = exp(-1j * 4 * alpha)
    oR5 = exp(2j * (-alpha + beta))
    oR6 = exp(2j * (alpha + beta))

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
    H2band = np.zeros([2 * q, 2 * q], dtype=complex)

    for m in range(0, q):

        H0[m, m] = E0[0, 0] + v2[0, 0] * exp(-4j * pi * m * eta) + v5[0, 0] * exp(4j * pi * m * eta)
        H0[m, pbc(m + 1, q)] = h2[0, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H0[m, pbc(m + 2, q)] = h1[0, 0] + o2[0, 0] * exp(-4j * pi * (m + 1) * eta) + o6[0, 0] * exp(4j * pi * (m + 1) * eta)
        H0[m, pbc(m + 3, q)] = v1[0, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H0[m, pbc(m + 4, q)] = o1[0, 0]
        H0[m, pbc(m - 1, q)] = h5[0, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H0[m, pbc(m - 2, q)] = h4[0, 0] + o3[0, 0] * exp(-4j * pi * (m - 1) * eta) + o5[0, 0] * exp(4j * pi * (m - 1) * eta)
        H0[m, pbc(m - 3, q)] = v3[0, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H0[m, pbc(m - 4, q)] = o4[0, 0]

        H11[m, m] = E0[1, 1] + v2[1, 1] * exp(-4j * pi * m * eta) + v5[1, 1] * exp(4j * pi * m * eta)
        H11[m, pbc(m + 1, q)] = h2[1, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H11[m, pbc(m + 2, q)] = h1[1, 1] + o2[1, 1] * exp(-4j * pi * (m + 1) * eta) + o6[1, 1] * exp(4j * pi * (m + 1) * eta)
        H11[m, pbc(m + 3, q)] = v1[1, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H11[m, pbc(m + 4, q)] = o1[1, 1]
        H11[m, pbc(m - 1, q)] = h5[1, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H11[m, pbc(m - 2, q)] = h4[1, 1] + o3[1, 1] * exp(-4j * pi * (m - 1) * eta) + o5[1, 1] * exp(4j * pi * (m - 1) * eta)
        H11[m, pbc(m - 3, q)] = v3[1, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H11[m, pbc(m - 4, q)] = o4[1, 1]

        H12T[m, m] = E0[2, 1] + v2[2, 1] * exp(4j * pi * m * eta) + v5[2, 1] * exp(-4j * pi * m * eta)
        H12T[m, pbc(m + 1, q)] = h2[2, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H12T[m, pbc(m + 2, q)] = h1[2, 1] + o2[2, 1] * exp(4j * pi * (m + 1) * eta) + o6[2, 1] * exp(-4j * pi * (m + 1) * eta)
        H12T[m, pbc(m + 3, q)] = v1[2, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H12T[m, pbc(m + 4, q)] = o1[2, 1]
        H12T[m, pbc(m - 1, q)] = h5[2, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H12T[m, pbc(m - 2, q)] = h4[2, 1] + o3[2, 1] * exp(4j * pi * (m - 1) * eta) + o5[2, 1] * exp(-4j * pi * (m - 1) * eta)
        H12T[m, pbc(m - 3, q)] = v3[2, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H12T[m, pbc(m - 4, q)] = o4[2, 1]

        H12[m, m] = E0[1, 2] + v2[1, 2] * exp(-4j * pi * m * eta) + v5[1, 2] * exp(4j * pi * m * eta)
        H12[m, pbc(m + 1, q)] = h2[1, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H12[m, pbc(m + 2, q)] = h1[1, 2] + o2[1, 2] * exp(-4j * pi * (m + 1) * eta) + o6[1, 2] * exp(4j * pi * (m + 1) * eta)
        H12[m, pbc(m + 3, q)] = v1[1, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H12[m, pbc(m + 4, q)] = o1[1, 2]
        H12[m, pbc(m - 1, q)] = h5[1, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H12[m, pbc(m - 2, q)] = h4[1, 2] + o3[1, 2] * exp(-4j * pi * (m - 1) * eta) + o5[1, 2] * exp(4j * pi * (m - 1) * eta)
        H12[m, pbc(m - 3, q)] = v3[1, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H12[m, pbc(m - 4, q)] = o4[1, 2]

        H1T[m, m] = E0[1, 0] + v2[1, 0] * exp(4j * pi * m * eta) + v5[1, 0] * exp(-4j * pi * m * eta)
        H1T[m, pbc(m + 1, q)] = h2[1, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[1, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H1T[m, pbc(m + 2, q)] = h1[1, 0] + o2[1, 0] * exp(4j * pi * (m + 1) * eta) + o6[1, 0] * exp(-4j * pi * (m + 1) * eta)
        H1T[m, pbc(m + 3, q)] = v1[1, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[1, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H1T[m, pbc(m + 4, q)] = o1[1, 0]
        H1T[m, pbc(m - 1, q)] = h5[1, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[1, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H1T[m, pbc(m - 2, q)] = h4[1, 0] + o3[1, 0] * exp(4j * pi * (m - 1) * eta) + o5[1, 0] * exp(-4j * pi * (m - 1) * eta)
        H1T[m, pbc(m - 3, q)] = v3[1, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[1, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H1T[m, pbc(m - 4, q)] = o4[1, 0]

        H1[m, m] = E0[0, 1] + v2[0, 1] * exp(-4j * pi * m * eta) + v5[0, 1] * exp(4j * pi * m * eta)
        H1[m, pbc(m + 1, q)] = h2[0, 1] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 1] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H1[m, pbc(m + 2, q)] = h1[0, 1] + o2[0, 1] * exp(-4j * pi * (m + 1) * eta) + o6[0, 1] * exp(4j * pi * (m + 1) * eta)
        H1[m, pbc(m + 3, q)] = v1[0, 1] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 1] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H1[m, pbc(m + 4, q)] = o1[0, 1]
        H1[m, pbc(m - 1, q)] = h5[0, 1] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 1] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H1[m, pbc(m - 2, q)] = h4[0, 1] + o3[0, 1] * exp(-4j * pi * (m - 1) * eta) + o5[0, 1] * exp(4j * pi * (m - 1) * eta)
        H1[m, pbc(m - 3, q)] = v3[0, 1] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 1] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H1[m, pbc(m - 4, q)] = o4[0, 1]

        H22[m, m] = E0[2, 2] + v2[2, 2] * exp(-4j * pi * m * eta) + v5[2, 2] * exp(4j * pi * m * eta)
        H22[m, pbc(m + 1, q)] = h2[2, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H22[m, pbc(m + 2, q)] = h1[2, 2] + o2[2, 2] * exp(-4j * pi * (m + 1) * eta) + o6[2, 2] * exp(4j * pi * (m + 1) * eta)
        H22[m, pbc(m + 3, q)] = v1[2, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H22[m, pbc(m + 4, q)] = o1[2, 2]
        H22[m, pbc(m - 1, q)] = h5[2, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H22[m, pbc(m - 2, q)] = h4[2, 2] + o3[2, 2] * exp(-4j * pi * (m - 1) * eta) + o5[2, 2] * exp(4j * pi * (m - 1) * eta)
        H22[m, pbc(m - 3, q)] = v3[2, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H22[m, pbc(m - 4, q)] = o4[2, 2]

        H2T[m, m] = E0[2, 0] + v2[2, 0] * exp(4j * pi * m * eta) + v5[2, 0] * exp(-4j * pi * m * eta)
        H2T[m, pbc(m + 1, q)] = h2[2, 0] * exp(1j * 2 * pi * (m + 1 / 2) * eta) + h6[2, 0] * exp(-1j * 2 * pi * (m + 1 / 2) * eta)
        H2T[m, pbc(m + 2, q)] = h1[2, 0] + o2[2, 0] * exp(4j * pi * (m + 1) * eta) + o6[2, 0] * exp(-4j * pi * (m + 1) * eta)
        H2T[m, pbc(m + 3, q)] = v1[2, 0] * exp(1j * 2 * pi * (m + 3 / 2) * eta) + v6[2, 0] * exp(-1j * 2 * pi * (m + 3 / 2) * eta)
        H2T[m, pbc(m + 4, q)] = o1[2, 0]
        H2T[m, pbc(m - 1, q)] = h5[2, 0] * exp(-1j * 2 * pi * (m - 1 / 2) * eta) + h3[2, 0] * exp(1j * 2 * pi * (m - 1 / 2) * eta)
        H2T[m, pbc(m - 2, q)] = h4[2, 0] + o3[2, 0] * exp(4j * pi * (m - 1) * eta) + o5[2, 0] * exp(-4j * pi * (m - 1) * eta)
        H2T[m, pbc(m - 3, q)] = v3[2, 0] * exp(1j * 2 * pi * (m - 3 / 2) * eta) + v4[2, 0] * exp(-1j * 2 * pi * (m - 3 / 2) * eta)
        H2T[m, pbc(m - 4, q)] = o4[2, 0]

        H2[m, m] = E0[0, 2] + v2[0, 2] * exp(-4j * pi * m * eta) + v5[0, 2] * exp(4j * pi * m * eta)
        H2[m, pbc(m + 1, q)] = h2[0, 2] * exp(-1j * 2 * pi * (m + 1 / 2) * eta) + h6[0, 2] * exp(1j * 2 * pi * (m + 1 / 2) * eta)
        H2[m, pbc(m + 2, q)] = h1[0, 2] + o2[0, 2] * exp(-4j * pi * (m + 1) * eta) + o6[0, 2] * exp(4j * pi * (m + 1) * eta)
        H2[m, pbc(m + 3, q)] = v1[0, 2] * exp(-1j * 2 * pi * (m + 3 / 2) * eta) + v6[0, 2] * exp(1j * 2 * pi * (m + 3 / 2) * eta)
        H2[m, pbc(m + 4, q)] = o1[0, 2]
        H2[m, pbc(m - 1, q)] = h5[0, 2] * exp(1j * 2 * pi * (m - 1 / 2) * eta) + h3[0, 2] * exp(-1j * 2 * pi * (m - 1 / 2) * eta)
        H2[m, pbc(m - 2, q)] = h4[0, 2] + o3[0, 2] * exp(-4j * pi * (m - 1) * eta) + o5[0, 2] * exp(4j * pi * (m - 1) * eta)
        H2[m, pbc(m - 3, q)] = v3[0, 2] * exp(-1j * 2 * pi * (m - 3 / 2) * eta) + v4[0, 2] * exp(1j * 2 * pi * (m - 3 / 2) * eta)
        H2[m, pbc(m - 4, q)] = o4[0, 2]

    H[0:q, 0:q] = H0
    H[0:q, q : 2 * q] = H1
    H[0:q, 2 * q : 3 * q] = H2
    H[q : 2 * q, 0:q] = H1T
    H[q : 2 * q, q : 2 * q] = H11
    H[q : 2 * q, 2 * q : 3 * q] = H12
    H[2 * q : 3 * q, 0:q] = H2T
    H[2 * q : 3 * q, q : 2 * q] = H12T
    H[2 * q : 3 * q, 2 * q : 3 * q] = H22

    return H
