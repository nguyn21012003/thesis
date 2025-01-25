import numpy as np
import csv
from numpy import pi, identity
from numpy import linalg as LA
from scipy import constants
from math import sqrt, cos, sin
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os


hbar = constants.hbar
B = 10 * 0
e = 9.1e-31
N = 100
lambd = 0.073


def para(argument):
    argument = int(argument)
    matt = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]
    a = [3.190, 3.191, 3.326, 3.325, 3.357, 3.560]
    e1 = [0.820, 0.717, 0.684, 0.728, 0.588, 0.697]
    e2 = [1.931]
    t0 = [-0.176]
    t1 = [-0.101]
    t2 = [0.531]
    t11 = [0.084]
    t12 = [0.169]
    t22 = [0.07]
    r0 = [0.070, 0.069, 0.039, 0.036, 0.003, -0.015]
    r1 = [-0.252, -0.261, -0.209, -0.234, -0.025, -0.209]
    r2 = [0.084, 0.107, 0.069, 0.107, -0.169, 0.107]
    r11 = [0.019, -0.003, 0.052, 0.044, 0.082, 0.115]
    r12 = [0.093, 0.109, 0.060, 0.075, 0.051, 0.009]
    u0 = [-0.043]
    u1 = [0.047, 0.045, 0.036, 0.032, 0.103, 0.011]
    u2 = [0.005, 0.002, 0.008, 0.007, 0.187, -0.013]
    u11 = [0.304, 0.325, 0.272, 0.329, -0.045, 0.312]
    u12 = [-0.192, -0.206, -0.172, -0.202, -0.141, -0.177]
    u22 = [-0.162, -0.163, -0.15, -0.164, 0.087, -0.312]
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
                r0[argument],
                r1[argument],
                r2[argument],
                r11[argument],
                r12[argument],
                u0[argument],
                u1[argument],
                u2[argument],
                u11[argument],
                u12[argument],
                u22[argument],
            )


def eigenvalue(argument):
    matt, a_lattice, e1, e2, t0, t1, t2, t11, t12, t22, r0, r1, r2, r11, r12, u0, u1, u2, u11, u12, u22 = para(argument)
    n = (e / (8 * hbar)) * B * a_lattice**2 * sqrt(3)
    eigenvalues = []

    L1 = np.zeros((N, N))
    L2 = np.zeros((N, N))
    L3 = np.zeros((N, N))
    L4 = np.zeros((N, N))
    L5 = np.zeros((N, N))
    L6 = np.zeros((N, N))

    kx = np.zeros((N, N))
    ky = np.zeros((N, N))
    dki = 4 * pi * sqrt(3) / (3 * (N - 1) * a_lattice)
    dkj = 4 * pi * sqrt(3) / (3 * (N - 1) * a_lattice)

    ############################
    I2 = identity(2)
    I3 = identity(3)
    pauliZ = np.array([[1, 0], [0, -1]])
    #############################

    Lz = np.array([[0, 0, 0], [0, 0, 2j], [0, -2j, 0]])
    H_soc = lambd / 2 * np.kron(pauliZ, Lz)
    #############################
    g_fact = 0
    H_z = g_fact * np.kron(I3, pauliZ)
    #############################
    for i in range(N):
        for j in range(N):
            kx[i][j] = cos(pi / 6) * (i * dki + j * dkj) - 2 * pi / (a_lattice)
            ky[i][j] = sin(pi / 6) * (-i * dki + j * dkj) * 0 + 0
            # ky[i][j] = 0
            a = kx[i][j] * a_lattice / 2
            b = sqrt(3) / 2 * ky[i][j] * a_lattice

            V0 = (
                2 * t0 * (cos(2 * a) + 2 * cos(a) * cos(b))
                + 2 * r0 * (2 * cos(3 * a) * cos(b) + cos(2 * b))
                + 2 * u0 * (2 * cos(2 * a) * cos(2 * b) + cos(4 * a))
                + e1
            )

            V1 = (
                -2 * sqrt(3) * t2 * sin(a) * sin(b)
                + 2 * (r1 + r2) * sin(3 * a) * sin(b)
                - 2 * sqrt(3) * u2 * sin(2 * a) * sin(2 * b)
                + 1j
                * (
                    2 * t1 * sin(a) * (2 * cos(a) + cos(b))
                    + 2 * (r1 - r2) * sin(3 * a) * cos(b)
                    + 2 * u1 * sin(2 * a) * (2 * cos(2 * a) + cos(2 * b))
                )
            )
            V2 = (
                2 * t2 * (cos(2 * a) - cos(a) * cos(b))
                - 2 / sqrt(3) * (r1 + r2) * (cos(3 * a) * cos(b) - cos(2 * b))
                + 2 * u2 * (cos(4 * a) - cos(2 * a) * cos(2 * b))
                + 1j
                * (
                    2 * sqrt(3) * t1 * cos(a) * sin(b)
                    + 2 / sqrt(3) * sin(b) * (r1 - r2) * (cos(3 * a) + 2 * cos(b))
                    + 2 * sqrt(3) * u1 * cos(2 * a) * sin(2 * b)
                )
            )
            V11 = (
                (t11 + 3 * t22) * cos(a) * cos(b)
                + 2 * t11 * cos(2 * a)
                + 4 * r11 * cos(3 * a) * cos(b)
                + 2 * (r11 + sqrt(3) * r12) * cos(2 * b)
                + (u11 + 3 * u22) * cos(2 * a) * cos(2 * b)
                + 2 * u11 * cos(4 * a)
                + e2
            )
            V22 = (
                (t22 + 3 * t11) * cos(a) * cos(b)
                + 2 * t22 * cos(2 * a)
                + 2 * r11 * (2 * cos(3 * a) * cos(b) + cos(2 * b))
                + 2 / sqrt(3) * r12 * (4 * cos(3 * a) * cos(b) - cos(2 * b))
                + (3 * u11 + u22) * cos(2 * a) * cos(2 * b)
                + 2 * u22 * cos(4 * a)
                + e2
            )
            V12 = (
                1j * (4 * t12 * sin(a) * (cos(a) - cos(b)) + 4 * u12 * sin(2 * a) * (cos(2 * a) - cos(2 * b)))
                + sqrt(3) * (t22 - t11) * sin(a) * sin(b)
                + 4 * r12 * sin(3 * a) * sin(b)
                + sqrt(3) * (u22 - u11) * sin(2 * a) * sin(2 * b)
            )

            V1dag = np.conjugate(V1)
            V2dag = np.conjugate(V2)
            V12dag = np.conjugate(V12)

            H_tnn = np.array([[V0, V1, V2], [V1dag, V11, V12], [V2dag, V12dag, V22]])

            H_tnn6x6 = np.kron(I2, H_tnn)

            H = H_tnn6x6 + H_z + H_soc

            w, v = LA.eigh(H)

            eigenvalues.append(w)

            L1[i, j] = float(w[0])
            L2[i, j] = float(w[1])
            L3[i, j] = float(w[2])
            L4[i, j] = float(w[3])
            L5[i, j] = float(w[4])
            L6[i, j] = float(w[5])

    fig = plt.figure(figsize=(16, 8))

    # ax = fig.add_subplot(121, projection="3d")
    # ax.plot_surface(kx, ky, L1, cmap="viridis")
    # ax.plot_surface(kx, ky, L2, cmap="plasma")
    # ax.plot_surface(kx, ky, L3, cmap="inferno")
    # ax.plot_surface(kx, ky, L4, cmap="inferno")
    # ax.plot_surface(kx, ky, L5, cmap="inferno")
    # ax.plot_surface(kx, ky, L6, cmap="inferno")

    hex_code0 = "#000000"
    hex_code = None

    # bplt.subplot(1, 3, 3)
    plt.plot(kx, L1, color=hex_code0)
    plt.plot(kx, L2, color=hex_code0)
    plt.plot(kx, L3, color=hex_code0)
    plt.plot(kx, L4, color=hex_code0)
    plt.plot(kx, L5, color=hex_code0)
    plt.plot(kx, L6, color=hex_code0)

    plt.grid(True, axis="x")
    plt.gca().set_xticks([-4 * pi / (3 * a), 0, 4 * pi / (3 * a)], minor=True)
    plt.xlabel("kx")
    plt.xticks([-2 * pi / a, -4 * pi / (3 * a), 0, 4 * pi / (3 * a), 2 * pi / a], ["M", "K", "Γ", "K", "M"])

    plt.ylabel("eV")
    plt.yticks([-1, 0, 1, 2, 3, 4])

    plt.title(f"{matt}")
    plt.show()

    with open("HTNNSOC.txt", "w", newline="") as writefile:
        header = ["Lambda1", "Lambda2", "Lambda3", "Lambda4", "Lambda5", "Lambda6"]
        writer = csv.DictWriter(writefile, fieldnames=header)
        writer.writeheader()
        for i in range(len(L1)):
            for j in range(len(L1)):
                writer.writerow(
                    {"Lambda1": L1[i][j], "Lambda2": L2[i][j], "Lambda3": L3[i][j], "Lambda4": L4[i][j], "Lambda5": L5[i][j], "Lambda6": L6[i][j]}
                )


if __name__ == "__main__":
    start = time.time()
    eigenvalue(int(input()))
    # print(para(input()))
    end = time.time()
    print(end - start, "time")
