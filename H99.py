import numpy as np
import csv
from numpy import pi, exp
from numpy import linalg as LA
from math import sqrt, cos, sin
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

np.set_printoptions(precision=10, linewidth=1200, edgeitems=24)

e1 = 1.046
e2 = 2.104
t0 = -0.184
t1 = 0.401
t2 = 0.507
t11 = 0.218
t12 = 0.338
t22 = 0.057
a = 0.3190


def Hamiltonian(p, q, kx, ky):

    alpha = p / q
    h0 = np.zeros([q, q], dtype=complex)
    h1 = np.zeros([q, q], dtype=complex)
    h2 = np.zeros([q, q], dtype=complex)
    h11 = np.zeros([q, q], dtype=complex)
    h22 = np.zeros([q, q], dtype=complex)
    h12 = np.zeros([q, q], dtype=complex)

    for m in range(0, q):
        h0[m][m] = 0
        h1[m][m] = 0
        h2[m][m] = 0
        h11[m][m] = 0
        h22[m][m] = 0
        h12[m][m] = 0
        if m == 0:  #### First row

            h0[m][m + 1] = t0 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h0[m][q - 1] = t0 * np.exp(-q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h1[m][m + 1] = (t1 + sqrt(3) * t2) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h1[m][q - 1] = -(t1 + sqrt(3) * t2) / 2 * np.exp(-q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h2[m][m + 1] = (sqrt(3) * t1 - t2) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h2[m][q - 1] = (sqrt(3) * t1 - t2) / 2 * np.exp(-q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h11[m][m + 1] = (t11 + 3 * t22) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h11[m][q - 1] = (t11 + 3 * t22) / 2 * np.exp(-q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h22[m][m + 1] = (3 * t11 + t22) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h22[m][q - 1] = (3 * t11 + t22) * np.exp(-q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h12[m][m + 1] = sqrt(3) / 4 * (t11 - t22 - 4 * t12) * cos(2 * pi * alpha * (m + 1) - ky * a)
            h12[m][q - 1] = sqrt(3) / 4 * (t11 - t22 - 4 * t12) * np.exp(-q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

        elif m == q - 1:  ### last Row

            h0[m][m - 1] = t0 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h0[q - 1][0] = t0 * np.exp(q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h1[m][m - 1] = -(t1 + sqrt(3) * t2) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h1[q - 1][0] = (t1 + sqrt(3) * t2) / 2 * np.exp(q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h2[m][m - 1] = (sqrt(3) * t1 - t2) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h2[q - 1][0] = (sqrt(3) * t1 - t2) / 2 * np.exp(q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h11[m][m - 1] = (t11 + 3 * t22) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h11[q - 1][0] = (t11 + 3 * t22) / 2 * np.exp(q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h22[m][m - 1] = (3 * t11 + t22) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h22[q - 1][0] = (3 * t11 + t22) / 2 * np.exp(q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

            h12[m][m - 1] = (sqrt(3) / 4) * (t11 - t22 - 4 * t12) * cos(2 * pi * alpha * (m + 1) - ky * a)
            h12[q - 1][0] = (sqrt(3) / 4) * (t11 - t22 - 4 * t12) * np.exp(q * 1.0j * kx * a) * cos(2 * pi * (m + 1) * alpha - ky * a)

        else:

            h0[m][m - 1] = t0 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h0[m][m + 1] = t0 * cos(2 * pi * alpha * (m + 1) - ky * a)

            h1[m][m - 1] = -(t1 + sqrt(3) * t2) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h1[m][m + 1] = (t1 + sqrt(3) * t2) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)

            h2[m][m - 1] = (sqrt(3) * t1 - t2) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h2[m][m + 1] = (sqrt(3) * t1 - t2) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)

            h11[m][m - 1] = (t11 + 3 * t22) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h11[m][m + 1] = (t11 + 3 * t22) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)

            h22[m][m - 1] = (3 * t11 + t22) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)
            h22[m][m + 1] = (3 * t11 + t22) / 2 * cos(2 * pi * alpha * (m + 1) - ky * a)

            h12[m][m - 1] = (sqrt(3) / 4) * (t11 - t22 - 4 * t12) * cos(2 * pi * alpha * (m + 1) - ky * a)
            h12[m][m + 1] = (sqrt(3) / 4) * (t11 - t22 - 4 * t12) * cos(2 * pi * alpha * (m + 1) - ky * a)

    h0[0][q - 2] = t0 * np.exp(-q * 1.0j * kx * a)
    h0[q - 2][0] = t0 * np.exp(q * 1.0j * kx * a)

    h1[0][q - 2] = -t1 * np.exp(-q * 1.0j * kx * a)
    h1[q - 2][0] = t1 * np.exp(q * 1.0j * kx * a)

    h2[0][q - 2] = t2 * np.exp(-q * 1.0j * kx * a)
    h2[q - 2][0] = t2 * np.exp(q * 1.0j * kx * a)

    h11[0][q - 2] = t11 * np.exp(-q * 1.0j * kx * a)
    h11[q - 2][0] = t11 * np.exp(q * 1.0j * kx * a)

    h22[0][q - 2] = t22 * np.exp(-q * 1.0j * kx * a)
    h22[q - 2][0] = t22 * np.exp(q * 1.0j * kx * a)

    h12[0][q - 2] = t12 * np.exp(-q * 1.0j * kx * a)
    h12[q - 2][0] = t12 * np.exp(q * 1.0j * kx * a)

    h0[np.eye(q, k=2, dtype=bool)] = t0
    h0[np.eye(q, k=-2, dtype=bool)] = t0

    h1[np.eye(q, k=2, dtype=bool)] = t1
    h1[np.eye(q, k=-2, dtype=bool)] = t1

    h2[np.eye(q, k=2, dtype=bool)] = t2
    h2[np.eye(q, k=-2, dtype=bool)] = t2

    h11[np.eye(q, k=2, dtype=bool)] = t11
    h11[np.eye(q, k=-2, dtype=bool)] = t11

    h22[np.eye(q, k=2, dtype=bool)] = t22
    h22[np.eye(q, k=-2, dtype=bool)] = t22

    h12[np.eye(q, k=2, dtype=bool)] = t12
    h12[np.eye(q, k=-2, dtype=bool)] = t12

    H = np.zeros((3 * q, 3 * q), dtype=complex)

    H[0:q, 0:q] = h0

    H[0:q, q : 2 * q] = h1

    H[0:q, 2 * q : 3 * q] = h2

    H[q : 2 * q, 0:q] = np.conjugate(h1)

    H[q : 2 * q, q : 2 * q] = h11

    H[q : 2 * q, 2 * q : 3 * q] = h12

    H[2 * q : 3 * q, 0:q] = np.conjugate(h2)

    H[2 * q : 3 * q, q : 2 * q] = np.conjugate(h12)

    H[2 * q : 3 * q, 2 * q : 3 * q] = h22

    return H


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def main():

    plt.figure(figsize=(10, 10))
    qmax = 40
    with open("dataHofstadterButterfly.csv", "w", newline="") as writefile:
        header = [f"{'p':^10}", f"{'q':^10}", f"{'p/q':^20}", f"{'y':^20}", "eigenvalue"]
        writer = csv.DictWriter(writefile, fieldnames=header, delimiter="\t")
        writer.writeheader()
        for p in tqdm(range(1, qmax + 1)):
            for q in range(1, qmax + 1):
                if q > p:
                    if gcd(p, q) == 1:
                        alpha = p / q
                        y = np.zeros(3 * q)
                        y[:] = alpha

                        # print(H, "\n")
                        eigenvalue2 = np.linalg.eigvalsh(Hamiltonian(p, q, kx=4 * pi / (3 * a * q), ky=0))
                        eigenvalue1 = np.linalg.eigvalsh(Hamiltonian(p, q, kx=0, ky=0))
                        writer.writerow(
                            {
                                f"{'p':^10}": f"{p:^10}",
                                f"{'q':^10}": f"{q:^10}",
                                f"{'p/q':^20}": f"{p:>10}/{q:<10}",
                                f"{'y':^20}": f"{alpha:^20}",
                                "eigenvalue": eigenvalue2,
                            }
                        )
                        for i in range(len(eigenvalue2)):
                            plt.plot(y[:2], [eigenvalue1[i], eigenvalue2[i]], "-", c="red", markersize=0.1)

                        plt.plot(y, eigenvalue1, "o", c="red", markersize=0.3)
                        plt.plot(y, eigenvalue2, "o", c="red", markersize=0.3)

    plt.show()


if __name__ == "__main__":
    main()
