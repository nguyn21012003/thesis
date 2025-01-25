import numpy as np
from numpy import pi, linalg
from scipy import constants
from math import sqrt, cos, sin
import matplotlib as mpl
import matplotlib.pyplot as plt


# Define the constants: MoS2
e1 = 1.046
e2 = 2.104
t0 = 0
t1 = 0.401
t2 = 0.507
t11 = 0
t12 = 0.338
t22 = 0
a_lattice = 0.3190  # nm
hbar = 0.658
B = 10
eB = 10e-3 * B
m = 5.68
n = (eB / (8 * hbar)) * a_lattice**2 * sqrt(3)
N = 100
g_fact = eB * hbar / (2 * m)


def deter(arr):
    # Use numpy to calculate determinant for efficiency
    return np.linalg.det(arr)


k_x = np.zeros((N, N))
k_y = np.zeros((N, N))

dk_i = (4 * pi) / ((sqrt(3) * a_lattice) * (N - 1))
dk_j = (4 * pi) / ((sqrt(3) * a_lattice) * (N - 1))

print(f"{'Lambda 1':^20} | {'Lambda 2':^20} | {'Lambda 3':^20}")


def eigenvalue():
    lamda1 = np.zeros((N, N))
    lamda2 = np.zeros((N, N))
    lamda3 = np.zeros((N, N))
    lamda4 = np.zeros((N, N))

    for i in range(N):
        for k in range(N):
            k_x[i][k] = cos(pi / 6) * (i * dk_i + k * dk_j) - 2 * pi / a_lattice
            k_y[i][k] = sin(pi / 6) * (-i * dk_i + k * dk_j) * 0

            a = (k_x[i][k] * a_lattice) / 2
            b = (sqrt(3) * k_y[i][k] * a_lattice) / 2

            h0 = 2 * t0 * (cos(2 * a) + 2 * cos(n) * cos(a) * cos(b) - 2j * sin(n) * sin(a) * sin(b)) + e1

            h1 = 2j * t1 * (sin(2 * a) + cos(n) * sin(a) * cos(b) + 1j * sin(n) * cos(a) * sin(b)) - 2 * sqrt(3) * t2 * (
                cos(n) * sin(a) * sin(b) + 1j * sin(n) * cos(a) * cos(b)
            )

            h2 = 2 * t2 * (cos(2 * a) - cos(n) * cos(a) * cos(b) - 1j * sin(n) * sin(a) * sin(b)) + 2j * sqrt(3) * t1 * (
                cos(n) * cos(a) * sin(b) + 1j * sin(n) * sin(a) * cos(b)
            )

            h11 = (t11 + 3 * t22) * (cos(n) * cos(a) * cos(b) - 1j * sin(n) * sin(a) * sin(b)) + 2 * t11 * cos(2 * a) + e2

            h22 = (3 * t11 + t22) * (cos(n) * cos(a) * cos(b) - 1j * sin(n) * sin(a) * sin(b)) + 2 * t22 * cos(2 * a) + e2

            h12 = (
                4j * t12 * (sin(a) * cos(a))
                - 4 * t12 * (cos(n) * sin(a) * cos(b) - 1j * sin(n) * cos(a) * sin(b))
                + sqrt(3) * (t11 - t22) * (cos(n) * sin(a) * sin(b) + 1j * sin(n) * cos(a) * cos(b))
            )

            h1dag = np.conjugate(h1)
            h2dag = np.conjugate(h2)
            h12dag = np.conjugate(h12)

            H = np.array([[h0, h1, h2], [h1dag, h11, h12], [h2dag, h12dag, h22]])

            eigenvalues = np.linalg.eigvalsh(H)

            print(eigenvalues)

            # print(f"{lamda1[i, k]:^10} | {lamda2[i][k]:^10} | {lamda3[i][k]:^10}")

    plt.figure(figsize=(18, 6))
    plt.plot(k_x, lamda1, color="#000000")
    plt.plot(k_x, lamda2, color="#000000")
    plt.plot(k_x, lamda3, color="#000000")
    k_points = [-2 * pi / a_lattice, -4 * pi / (3 * a_lattice), 0, 2 * pi / a_lattice, 4 * pi / (3 * a_lattice)]
    k_labels = ["-M", "-K", "$\Gamma$", "M", "K"]
    plt.xticks(ticks=k_points, labels=k_labels)
    plt.grid(True)
    plt.xlabel("k-point")
    plt.ylabel("Energy (eV)")

    plt.show()


eigenvalue()
