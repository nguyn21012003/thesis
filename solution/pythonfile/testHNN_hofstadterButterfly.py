import numpy as np
import csv
from numpy import pi
from numpy import linalg as LA
from scipy import constants
from math import sqrt, cos, sin
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from tqdm.notebook import tqdm_notebook

hbar = 6.6e-1
B1 = 0
B2 = 0
eB1 = 10e-3 * B1
eB2 = 10e-3 * B2
N = 1000
me = 5.68
g_fact1 = eB1 * hbar / (2 * me)
g_fact2 = eB2 * hbar / (2 * me)
lambd = 0.073


def para(argument):
    argument = int(argument)
    matt = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]
    a = [3.190, 3.191, 3.326, 3.325, 3.357, 3.560]
    e1 = [1.046, 1.130, 0.919, 0.943, 0.605, 0.606]
    e2 = [2.104, 2.275, 2.065, 2.179, 1.972, 2.102]
    t0 = [0, -0.206, -0.188, -0.207, -0.169, -0.175]
    t1 = [0.401, 0.567, 0.317, 0.457, 0.228, 0.342]
    t2 = [0.507, 0.536, 0.456, 0.486, 0.390, 0.410]
    t11 = [0, 0.286, 0.211, 0.263, 0.207, 0.233]
    t12 = [0.338, 0.384, 0.290, 0.329, 0.239, 0.270]
    t22 = [0, -0.061, 0.130, 0.034, 0.252, 0.190]
    match argument:
        case argument:
            return (
                matt[argument],
                a[argument],
                e1[argument],
                e2[argument],
                t0[argument],
                t1[argument],
                t2[argument],
                t11[argument],
                t12[argument],
                t22[argument],
            )


def eigenvalue(q, argument):
    L1 = np.zeros([N, N])
    L2 = np.zeros([N, N])
    L3 = np.zeros([N, N])
    matt, a_lattice, e1, e2, t0, t1, t2, t11, t12, t22 = para(argument)
    # n = (eB1 / (8 * hbar)) * a_lattice**2 * sqrt(3)
    eigenvalues = []
    kx = np.zeros([N, N])
    ky = np.zeros([N, N])
    dki = 4 * pi * sqrt(3) / (3 * (N - 1) * a_lattice)
    dkj = 4 * pi * sqrt(3) / (3 * (N - 1) * a_lattice)
    alpha = p / q
    H = np.zeros([3 * q, 3 * q], dtype=complex)

    for i in tqdm(range(N), desc="Proccesing", unit="step", ascii=" #"):
        for j in range(N):
            kx[i][j] = cos(pi / 6) * (i * dki + j * dkj) - 2 * pi / (a_lattice)
            ky[i][j] = sin(pi / 6) * (-i * dki + j * dkj) * 0 + 0
            # ky[i][j] = 0
            a = kx[i][j] * a_lattice / 2
            b = sqrt(3) / 2 * ky[i][j] * a_lattice

            h0 = 2 * t0 * (cos(2 * a) + 2 * cos(alpha) * cos(a) * cos(b) + 2 * sin(alpha) * sin(a) * cos(b)) + e1
            h1 = 2j * t1 * (sin(2 * a) + cos(alpha) * sin(a) * cos(b) - sin(alpha) * cos(a) * cos(b)) - 2 * sqrt(3) * t2 * (
                cos(alpha) * sin(a) * sin(b) - sin(alpha) * cos(a) * sin(b)
            )
            h2 = 2 * t2 * (cos(2 * a) - cos(alpha) * cos(a) * cos(b) - sin(alpha) * sin(a) * cos(b)) + 2j * sqrt(3) * t1 * (
                cos(alpha) * cos(a) * sin(b) + sin(alpha) * sin(a) * sin(b)
            )
            h11 = (t11 + 3 * t22) * (cos(alpha) * cos(a) * cos(b) + sin(alpha) * sin(a) * cos(b)) + 2 * t11 * cos(2 * a) + e2
            h22 = (t22 + 3 * t11) * (cos(alpha) * cos(a) * cos(b) + sin(alpha) * sin(a) * cos(b)) + 2 * t22 * cos(2 * a) + e2
            h12 = 2j * t12 * (sin(a) * cos(a) - cos(alpha) * sin(a) * cos(b) + sin(alpha) * cos(a) * cos(b)) + sqrt(3) * (t22 - t11) * (
                cos(alpha) * sin(a) * sin(b) - sin(alpha) * cos(a) * sin(b)
            )
            h1dag = np.conjugate(h1)
            h2dag = np.conjugate(h2)
            h12dag = np.conjugate(h12)

            H = np.array([[h0, h1, h2], [h1dag, h11, h12], [h2dag, h12dag, h22]])
            w, v = LA.eigh(H)
            eigenvalues.append(w)
            L1[i, j] = float(w[0])
            L2[i, j] = float(w[1])
            L3[i, j] = float(w[2])
            print(H)

            # print(L1[i][j], kx[i][j], ky[i][j], i, j, "\n")


#    fig = plt.figure(figsize=(16, 8))
#
#    # ax = fig.add_subplot(121, projection="3d")
#    # ax.plot_surface(kx, ky, L1, cmap="viridis")
#    # ax.plot_surface(kx, ky, L2, cmap="plasma")
#    # ax.plot_surface(kx, ky, L3, cmap="inferno")
#
#    hex_code = "#000000"
#    hex_code1 = None
#    # plt.subplot(1, 3, 3)
#    plt.plot(kx, L1, color=hex_code)
#    plt.plot(kx, L2, color=hex_code)
#    plt.plot(kx, L3, color=hex_code)
#
#    plt.grid(True, axis="x")
#    plt.gca().set_xticks([-4 * pi / (3 * a), 0, 4 * pi / (3 * a)], minor=True)
#    plt.xlabel("kx")
#    plt.xticks([-2 * pi / a, -4 * pi / (3 * a), 0, 4 * pi / (3 * a), 2 * pi / a], ["M", "K", "Γ", "K", "M"])
#
#    plt.ylabel("eV")
#    plt.yticks([-1, 0, 1, 2, 3, 4])
#
#    plt.title(f"{matt}")
#    plt.show()
#
#    with open("eigenvalue_0.txt", "w", newline="") as writefile:
#        header = ["Lambda1", "Lambda2", "Lambda3"]
#        writer = csv.DictWriter(writefile, fieldnames=header)
#        writer.writeheader()
#        for i in range(len(L1)):
#            for j in range(len(L1)):
#                writer.writerow({"Lambda1": L1[i][j], "Lambda2": L2[i][j], "Lambda3": L3[i][j]})


if __name__ == "__main__":
    start = time.time()
    q_max = 2
    eigenvalue(q_max, int(input(f"Nhap vat lieu vo: ")))
    # print(para(input()))
    end = time.time()
    print(end - start, "time")