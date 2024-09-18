import numpy as np
import csv
from numpy import pi
from numpy import linalg as LA
from scipy import constants
from math import sqrt, cos, sin
import matplotlib as mpl
import matplotlib.pyplot as plt
import time, sys


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
L = 4 * pi * sqrt(3) / (3 * a_lattice)


def eigenvalue():
    eigenvalues = []
    H = np.zeros((3, 3), dtype=complex)
    Lamb1 = np.zeros((N, N))
    Lamb2 = np.zeros((N, N))
    Lamb3 = np.zeros((N, N))

    ak1 = np.zeros(N)
    ak2 = np.zeros(N)
    G = 4 * pi / (sqrt(3) * a_lattice)

    k1min = -G / 2
    k1max = G / 2
    k2min = -G / 2
    k2max = G / 2

    dk1 = (k1max - k1min) / (N - 1)
    dk2 = (k2max - k2min) / (N - 1)

    for i in range(N):
        ak1[i] = k1min + i * dk1

    for j in range(N):
        ak2[j] = k2min + j * dk2

    akx = np.zeros((len(ak1), len(ak2)))
    aky = np.zeros((len(ak1), len(ak2)))

    for i in range(N):
        for j in range(N):
            akx[i][j] = sqrt(3) / 2 * (ak1[i] + ak2[j])
            aky[i][j] = -1 / 2 * (ak1[i] - ak2[j])

            a = akx[i][j] / 2 * a_lattice
            b = sqrt(3) / 2 * aky[i][j] * a_lattice

            H[0][0] = 2 * t0 * (cos(2 * a) + 2 * cos(n) * cos(a) * cos(b) + 2 * sin(n) * sin(a) * cos(b)) + e1
            H[0][1] = 2j * t1 * (sin(2 * a) + cos(n) * sin(a) * cos(b) - sin(n) * cos(a) * cos(b)) - 2 * sqrt(3) * t2 * (
                cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
            )
            H[0][2] = 2 * t2 * (cos(2 * a) - cos(n) * cos(a) * cos(b) - sin(n) * sin(a) * cos(b)) + 2j * sqrt(3) * t1 * (
                cos(n) * cos(a) * sin(b) + sin(n) * sin(a) * sin(b)
            )
            H[1][1] = (t11 + 3 * t22) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t11 * cos(2 * a) + e2
            H[2][2] = (t22 + 3 * t11) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t22 * cos(2 * a) + e2
            H[1][2] = 4j * t12 * (sin(a) * cos(a) - cos(n) * sin(a) * cos(b) + sin(n) * cos(a) * cos(b)) + sqrt(3) * (t22 - t11) * (
                cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
            )
            H[1][0] = np.conjugate(H[0][1])
            H[2][0] = np.conjugate(H[0][2])
            H[2][1] = np.conjugate(H[1][2])

            # fk = L**2 * sqrt(3) * dki * dkj * H / (8 * pi**2)
            w, _ = LA.eigh(H)
            eigenvalues.append(w)
            Lamb1[i, j] = float(w[0])
            Lamb2[i, j] = float(w[1])
            Lamb3[i, j] = float(w[2])

    print(akx[0][0])
    print(akx[8][0])
    print(aky[1][0])
    print(aky[8][8])
    if akx[0][0] == sqrt(3) / 2 * (k2min + k1min):
        print(True)
    # np.savetxt("e.txt", Lamb1, header="1")
    fig = plt.figure(figsize=(16, 8))
    plt.subplot(1, 3, 1)
    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(aky, akx, Lamb1, cmap="viridis")
    ax.plot_surface(aky, akx, Lamb2, cmap="plasma")
    ax.plot_surface(aky, akx, Lamb3, cmap="inferno")

    plt.subplot(1, 3, 3)
    plt.plot(akx, Lamb1)
    plt.plot(akx, Lamb2)
    plt.plot(akx, Lamb3)

    plt.grid(True)
    plt.xlabel("kx")
    plt.ylabel("ky")

    plt.show()


if __name__ == "__main__":
    start = time.time()
    a = eigenvalue()
    end = time.time()
    timing = end - start
    print(f"{timing}sscond")
