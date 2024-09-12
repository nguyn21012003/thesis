import numpy as np
import csv
from numpy.linalg import eig
from scipy import constants
from math import sqrt, e, cos, sin


def eigenvalue():
    c = 3.190
    hbar = constants.hbar
    B = 0
    n = 0

    kx = np.linspace(0, 18, num=18)
    ky = np.linspace(0, 18, num=18)

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

    for kx_val in kx:
        for ky_val in ky:
            a = 1 / 2 * kx_val * c
            b = sqrt(3) / 2 * ky_val * c

            h0 = 2 * t0 * (cos(2 * a) + 2 * cos(n) * cos(a) * cos(b) + 2 * sin(n) * sin(a) * cos(b)) + e1
            h1 = 2j * t1 * (sin(2 * a) + cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * cos(b)) - 2 * sqrt(3) * t2
            h2 = 2 * t2 * (cos(2 * a) - cos(n) * cos(a) * cos(b) - sin(n) * sin(a) * sin(b)) + 2j * sqrt(3) * t1 * (
                cos(n) * cos(a) * cos(b) + 1j * sin(n) * sin(a) * sin(b)
            )
            h11 = (t11 + 3 * t22) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t11 * cos(2 * a) + e2
            h22 = (t22 + 3 * t11) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t22 * cos(2 * a) + e2
            h12 = 4j * t12 * (sin(a) * cos(a) - cos(n) * sin(a) * cos(b) + sin(n) * cos(a) * cos(b)) + sqrt(3) * (t22 - t11) * (
                cos(n) * sin(a) * sin(b) + sin(n) * cos(a) * sin(b)
            )

            row1 = [h0, h1, h2]
            row2 = [np.conjugate(h1), h11, h12]
            row3 = [np.conjugate(h2), np.conjugate(h12), h22]
            H = np.array([row1, row2, row3])

            w, v = eig(H)

            eigenvalues.append(w)

    with open("file.csv", "w", newline="") as writefile:
        writer = csv.writer(writefile)
        writer.writerows(eigenvalues)


def main():
    print(eigenvalue())


if __name__ == "__main__":
    main()
