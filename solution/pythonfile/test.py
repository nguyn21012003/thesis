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
    eigenvectors = []
    kx = np.zeros(N)
    ky = np.zeros(N)
    for i in range(0, N):
        dk = 4 * pi / (a_lattice * (N + 1))
        kx[0] = -2 * pi / a_lattice
        if i + 1 < N:
            kx[i + 1] = kx[i] + dk
        ky[i] = 0

        a = 1 / 2 * kx[i] * a_lattice
        b = sqrt(3) / 2 * ky[i] * a_lattice

        h0 = 2 * t0 * (cos(2 * a) + 2 * cos(n) * cos(a) * cos(b) + 2 * sin(n) * sin(a) * cos(b)) + e1
        h1 = 2j * t1 * (sin(2 * a) + cos(n) * sin(a) * cos(b) - sin(n) * cos(a) * cos(b)) - 2 * sqrt(3) * t2 * (
            cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
        )
        h2 = 2 * t2 * (cos(2 * a) - cos(n) * cos(a) * cos(b) - sin(n) * sin(a) * cos(b)) + 2j * sqrt(3) * t1 * (
            cos(n) * cos(a) * sin(b) + sin(n) * sin(a) * sin(b)
        )
        h11 = (t11 + 3 * t22) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t11 * cos(2 * a) + e2
        h22 = (t22 + 3 * t11) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t22 * cos(2 * a) + e2
        h12 = 4j * t12 * (sin(a) * cos(a) - cos(n) * sin(a) * cos(b) + sin(n) * cos(a) * cos(b)) + sqrt(3) * (t22 - t11) * (
            cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
        )
        h1dag = np.conjugate(h1)
        h2dag = np.conjugate(h2)
        h12dag = np.conjugate(h12)

        H = np.array([[h0, h1, h2], [h1dag, h11, h12], [h2dag, h12dag, h22]])
        w = LA.eigvalsh(H)
        eigenvalues.append(w)

    with open("eigenvalue.txt", "w", newline="") as writefile:
        header = ["lambda1", "lambda2", "lambda3"]
        writer = csv.DictWriter(writefile, fieldnames=header)
        writer.writeheader()
        for i in range(len(eigenvalues)):
            writer.writerow({"lambda1": eigenvalues[i][0], "lambda2": eigenvalues[i][1], "lambda3": eigenvalues[i][2]})

    k_points = np.linspace(-2 * pi / a_lattice, 2 * pi / a_lattice, N)
    lambda1 = []
    lambda2 = []
    lambda3 = []
    for ev in eigenvalues:
        lambda1.append([ev[0]])
        lambda2.append([ev[1]])
        lambda3.append([ev[2]])

    energies = np.array([lambda1, lambda2, lambda3])
    plt.figure(figsize=(10, 6))
    for i in range(len(energies)):
        plt.plot(k_points, energies[i, :], label=f"Band {i + 1}")

    plt.xlabel("k-point")
    plt.ylabel("Energy (eV)")

    plt.legend()
    plt.show()


def main():
    start = time.time()
    eigenvalue()
    # plot("eigenvalue.txt")
    end = time.time()
    print(end - start, "time")


if __name__ == "__main__":
    main()
