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


def para(argument):
    argument = int(argument)
    matt = ["MoS2", "WS2", "MoSe2", "WSe2", "MoTe2", "WTe2"]
    a = [0.3190, 3.191, 3.326, 3.325, 3.357, 3.560]
    e1 = [1.046, 1.130, 0.919, 0.943, 0.605, 0.606]
    e2 = [2.104, 2.275, 2.065, 2.179, 1.972, 2.102]
    t0 = [-0.184, -0.206, -0.188, -0.207, -0.169, -0.175]
    t1 = [0.401, 0.567, 0.317, 0.457, 0.228, 0.342]
    t2 = [0.507, 0.536, 0.456, 0.486, 0.390, 0.410]
    t11 = [0.218, 0.286, 0.211, 0.263, 0.207, 0.233]
    t12 = [0.338, 0.384, 0.290, 0.329, 0.239, 0.270]
    t22 = [0.057, -0.061, 0.130, 0.034, 0.252, 0.190]
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


def Hamiltonian(argument, p, q, kx, ky):
    matt, a, e1, e2, t0, t1, t2, t11, t12, t22 = para(argument)
    alpha = p / q
    h0 = np.zeros([q, q], dtype=complex)
    h1 = np.zeros([q, q], dtype=complex)
    h2 = np.zeros([q, q], dtype=complex)
    h11 = np.zeros([q, q], dtype=complex)
    h22 = np.zeros([q, q], dtype=complex)
    h12 = np.zeros([q, q], dtype=complex)

    for m in range(0, q):

        h0[m][m] = e1
        h1[m][m] = 0 * (2 * t1 + 2 * sqrt(3) * t2) * cos(2 * pi * alpha * (m + 1) - ky * a)
        h2[m][m] = (2 * sqrt(3) * t1 - 2 * t2) * cos(2 * pi * alpha * (m + 1) - ky * a)
        h11[m][m] = e2
        h22[m][m] = e2
        h12[m][m] = sqrt(3) * (t11 - t22 - 4 * t12) * cos(2 * pi * alpha * (m + 1) - ky * a)

        if m == 0:  #### First row

            h0[m][m + 1] = t0  #### Thành phần ma trận thứ hai của dòng đầu tiên
            h0[m][q - 1] = t0 * (np.exp(-1j * kx * q * a))  # Thành phân ma trận cuối cùng của dòng đầu tiên

            h1[m][m + 1] = t1
            h1[m][q - 1] = -t1 * np.exp(-1j * kx * q * a)

            h2[m][m + 1] = t2
            h2[m][q - 1] = t2 * (np.exp(-1j * kx * q * a))

            h11[m][m + 1] = t11
            h11[m][q - 1] = t11 * (np.exp(-1j * kx * q * a))

            h22[m][m + 1] = t22
            h22[m][q - 1] = t22 * (np.exp(-1j * kx * q * a))

            h12[m][m + 1] = t12
            h12[m][q - 1] = t12 * (np.exp(-1j * kx * q * a))

        elif m == q - 1:  ### last Row

            h0[m][m - 1] = t0
            h0[q - 1][0] = t0 * np.exp(q * 1.0j * kx * a)

            h1[m][m - 1] = -t1
            h1[q - 1][0] = t1 * np.exp(q * 1.0j * kx * a)

            h2[m][m - 1] = t2
            h2[q - 1][0] = t2 * np.exp(q * 1.0j * kx * a)

            h11[m][m - 1] = t11
            h11[q - 1][0] = t11 * np.exp(q * 1.0j * kx * a)

            h22[m][m - 1] = t22
            h22[q - 1][0] = t22 * np.exp(q * 1.0j * kx * a)

            h12[m][m - 1] = t12
            h12[q - 1][0] = t12 * np.exp(q * 1.0j * kx * a)

        else:

            h0[m][m + 1] = t0  ## thành phần bên trái đường chéo
            h0[m][m - 1] = t0  ## thành phần bên phải đường chéo

            h1[m][m + 1] = t1
            h1[m][m - 1] = -t1

            h2[m][m + 1] = t2
            h2[m][m - 1] = t2

            h11[m][m + 1] = t11
            h11[m][m - 1] = t11

            h22[m][m + 1] = t22
            h22[m][m - 1] = t22

            h12[m][m + 1] = t12
            h12[m][m - 1] = t12

    H = np.zeros((3 * q, 3 * q), dtype=complex)

    # H[0:q, 0:q] = h0
    # H[0:q, q : 2 * q] = h0
    # H[0:q, 2 * q : 3 * q] = h2
    # H[q : 2 * q, 0:q] = np.conjugate(h1).T
    # H[q : 2 * q, q : 2 * q] = h11
    # H[q : 2 * q, 2 * q : 3 * q] = h12
    # H[2 * q : 3 * q, 0:q] = np.conjugate(h22).T
    # H[2 * q : 3 * q, q : 2 * q] = np.conjugate(h12).T
    # H[2 * q : 3 * q, 2 * q : 3 * q] = h22

    ############ 3 band
    # H[0:q, 0:q] = h0
    # H[0:q, q : 2 * q] = h0
    # H[0:q, 2 * q : 3 * q] = h0
    # H[q : 2 * q, 0:q] = np.conjugate(h0).T
    # H[q : 2 * q, q : 2 * q] = h0
    # H[q : 2 * q, 2 * q : 3 * q] = h0
    # H[2 * q : 3 * q, 0:q] = np.conjugate(h0).T
    # H[2 * q : 3 * q, q : 2 * q] = np.conjugate(h0).T
    # H[2 * q : 3 * q, 2 * q : 3 * q] = h0

    ############ 3 band
    H[0:q, 0:q] = h0
    # H[0:q, q : 2 * q] = h0
    # H[0:q, 2 * q : 3 * q] = h0
    # H[q : 2 * q, 0:q] = np.conjugate(h0).T
    H[q : 2 * q, q : 2 * q] = h0
    # H[q : 2 * q, 2 * q : 3 * q] = h0
    # H[2 * q : 3 * q, 0:q] = np.conjugate(h0).T
    # H[2 * q : 3 * q, q : 2 * q] = np.conjugate(h0).T
    H[2 * q : 3 * q, 2 * q : 3 * q] = h0

    return H


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def main():
    choice = 0  # int(input(("Input material: ")))
    qmax = 40  # int(input("Input the range q max aka the magnetic cell: "))
    matt, a, e1, e2, t0, t1, t2, t11, t12, t22 = para(choice)
    plt.figure(figsize=(7, 7))
    plt.title(f"{matt}")
    with open("dataHofstadterButterfly.csv", "w", newline="") as writefile:
        header = [
            f"{'p':^10}",
            f"{'q':^10}",
            f"{'p/q':^20}",
            f"{'y':^20}",
            "eigenvalue",
        ]
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
                        eigenvalue1 = np.linalg.eigvalsh(Hamiltonian(choice, p, q, kx=4 * pi / (3 * q), ky=0))
                        eigenvalue2 = np.linalg.eigvalsh(Hamiltonian(choice, p, q, kx=0, ky=0))

                        writer.writerow(
                            {
                                f"{'p':^10}": f"{p:^10}",
                                f"{'q':^10}": f"{q:^10}",
                                f"{'p/q':^20}": f"{p:>10}/{q:<10}",
                                f"{'y':^20}": f"{alpha:^20}",
                                "eigenvalue": eigenvalue1,
                            }
                        )
                        # for i in range(len(eigenvalue2)):
                        #    plt.plot([eigenvalue1[i], eigenvalue2[i]], y[:2], "-", c="red", markersize=0.1)
                        #
                        plt.plot(y, eigenvalue1, "o", c="red", markersize=0.1)
                        plt.plot(y, eigenvalue2, "o", c="red", markersize=0.1)

    plt.show()
    plt.savefig(f"{matt}_qmax_{qmax}.png")


if __name__ == "__main__":
    main()