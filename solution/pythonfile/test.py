import numpy as np
import csv
from numpy import pi
from numpy import linalg as LA
from scipy import constants
from math import sqrt, cos, sin
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time


def eigenvalue():
    a_lattice = 3.190
    hbar = constants.hbar
    B = 0
    e = 9.1e-31
    eta = (e / hbar) * B * a_lattice**2 * sqrt(3) / 8
    n = 6
    L = 1
    u1 = (2 * pi / (3 * a_lattice), 2 * pi / sqrt(3) * a_lattice)
    u2 = (2 * pi / (3 * a_lattice), -2 * pi / sqrt(3) * a_lattice)
    # v1 = n1 * u1 / n
    # v2 = n2 * u / n

    kx = np.zeros(n)
    ky = np.zeros(n)
    for n1 in range(n):
        for n2 in range(n):
            v1 = n1 * u1 / n
            v2 = n2 * u2 / n
            kx[n1] = (2 * v1 * pi + 2 * v2 * pi - 2 * pi) / a_lattice
            ky[n2] = (2 * sqrt(3) * v1 * pi - 2 * sqrt(3) * v2 * pi) / (3 * a_lattice)

    print(kx)


eigenvalue()
