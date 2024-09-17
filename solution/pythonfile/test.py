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
    a_lattice = 31.90
    hbar = constants.hbar
    B = 0
    e = 9.1e-31
    n = (e / hbar) * B * a_lattice**2 * sqrt(3) / 8
    N = 2000
    L = 1
    u1x = 2 * pi / (3 * a_lattice)
    u1y = 2 * pi / (sqrt(3) * a_lattice)
    u2x = 2 * pi / (sqrt(3) * a_lattice)
    u2y = -2 * pi / (sqrt(3) * a_lattice)
    kx = np.zeros(N)
    ky = np.zeros(N)
    for n in range(1, N):
        kx[0] = -2 * pi / a_lattice
        kx[n] = kx[0] + (n - 1) * 4 * pi / (a_lattice * N)
        ky[n] = 0

    # Hamiltonian parameters
    t0 = -0.184
    t1 = 0.401
    t2 = 0.507
    t11 = 0.218
    t12 = 0.338
    t22 = 0.057
    e1 = 1.046
    e2 = 2.104

    eigenvalues = []
    eigenvectors = []

    for kx_val in kx:
        for ky_val in ky:
            a = 1 / 2 * kx_val * a_lattice
            b = sqrt(3) / 2 * ky_val * a_lattice

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
            row1 = [h0, h1, h2]
            row2 = [h1dag, h11, h12]
            row3 = [h2dag, h12dag, h22]
            H = np.array([row1, row2, row3])

            w, v = LA.eigh(H)

            eigenvalues.append(w)

    with open("eigenvalue.txt", "w", newline="") as writefile:
        header = ["lambda1", "lambda2", "lambda3"]
        writer = csv.DictWriter(writefile, fieldnames=header)
        writer.writeheader()
        for i in range(len(eigenvalues)):
            writer.writerow({"lambda1": eigenvalues[i][0], "lambda2": eigenvalues[i][1], "lambda3": eigenvalues[i][2]})


def plot(filename):
    a_lattice = 31.90
    k_points = np.linspace(-2 * pi / a_lattice, 2 * pi / a_lattice, 2000**2)
    lambda1 = []
    lambda2 = []
    lambda3 = []
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            lambda1.append(float(row["lambda1"]))
            lambda2.append(float(row["lambda2"]))
            lambda3.append(float(row["lambda3"]))

    energies = np.array([lambda1, lambda2, lambda3])
    plt.figure(figsize=(10, 6))
    fine_k_points = np.linspace(k_points.min(), k_points.max(), 500)

    for i in range(energies.shape[0]):

        interpolator = interp1d(k_points, energies[i, :], kind="cubic")
        smooth_energies = interpolator(fine_k_points)

        plt.plot(fine_k_points, smooth_energies, label=f"Band {i+1}")
        plt.scatter(k_points, energies[i, :], marker="")

    plt.xlabel("k-point")
    plt.ylabel("Energy (eV)")

    plt.yticks([])
    plt.show()


def main():
    start = time.time()
    eigenvalue()
    plot("eigenvalue.txt")
    end = time.time()
    print(end - start, "time")


if __name__ == "__main__":
    main()
