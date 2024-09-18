import numpy as np
import csv
from numpy import pi
from numpy import linalg as LA
from scipy import constants
from math import sqrt, cos, sin
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


e1 = 1.238
e2 = 2.366
t0 = -0.218
t1 = 0.444
t2 = 0.533
t11 = 0.250
t12 = 0.360
t22 = 0.047

a_lattice = 3.129
hbar = constants.hbar
B = 0
e = 9.1e-31
n = (e / (8 * hbar)) * B * a_lattice**2 * sqrt(3)
N = 100


def eigenvalue():
    eigenvalues = []
    L1 = np.zeros((N, N))
    L2 = np.zeros((N, N))
    L3 = np.zeros((N, N))
    kx = np.zeros((N, N))
    ky = np.zeros((N, N))
    dki = 4 * pi * sqrt(3) / (3 * (N - 1) * a_lattice)
    dkj = 4 * pi * sqrt(3) / (3 * (N - 1) * a_lattice)
    for i in range(N):
        for j in range(N):
            kx[i][j] = cos(pi / 6) * (i * dki + j * dkj) - 2 * pi / (a_lattice)
            ky[i][j] = sin(pi / 6) * (-i * dki + j * dkj) + 0
            # ky[i][j] = 0
            a = kx[i][j] * a_lattice / 2
            b = sqrt(3) / 2 * ky[i][j] * a_lattice

            h0 = 2 * t0 * (cos(2 * a) + 2 * cos(a) * cos(b)) + e1
            h1 = 2j * t1 * (sin(2 * a) + sin(a) * cos(b)) - 2 * sqrt(3) * t2 * sin(a) * sin(b)
            h2 = 2 * t2 * (cos(2 * a) - cos(a) * cos(b)) + 2j * sqrt(3) * t1 * cos(a) * sin(b)
            h11 = (t11 + 3 * t22) * cos(a) * cos(b) + 2 * t11 * cos(2 * a) + e2
            h22 = (3 * t11 + t22) * cos(a) * cos(b) + 2 * t22 * cos(2 * a) + e2
            h12 = 4j * t12 * (sin(a) * cos(a) - sin(a) * cos(b)) + sqrt(3) * (t22 - t11) * sin(a) * sin(b)
            h1dag = np.conjugate(h1)
            h2dag = np.conjugate(h2)
            h12dag = np.conjugate(h12)

            H = np.array([[h0, h1, h2], [h1dag, h11, h12], [h2dag, h12dag, h22]])
            w, v = LA.eigh(H)
            eigenvalues.append(w)
            L1[i, j] = float(w[0])
            L2[i, j] = float(w[1])
            L3[i, j] = float(w[2])

            # print(L1[i][j], kx[i][j], ky[i][j], i, j, "\n")

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(kx, ky, L1, cmap="viridis")
    ax.plot_surface(kx, ky, L2, cmap="plasma")
    ax.plot_surface(kx, ky, L3, cmap="inferno")

    plt.subplot(1, 3, 3)
    plt.plot(kx, L1)
    plt.plot(kx, L2)
    plt.plot(kx, L3)

    plt.grid(True)
    plt.xlabel("kx")
    plt.ylabel("ky")

    plt.show()


if __name__ == "__main__":
    eigenvalue()
