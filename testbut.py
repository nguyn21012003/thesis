"""n = 20  # Tương ứng 3n
cols = 3 * n  # Số cột
rows = 3 * n  # Số hàng"""

import numpy as np
from math import cos, sin, pi, sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=10, linewidth=2000, edgeitems=40)
# Các hằng số
e1 = 1.046
e2 = 2.104
t0 = -0.184
t1 = 0.401
t2 = 0.507
t11 = 0.218
t12 = 0.338
t22 = 0.057


def tridiagonal_matrix(listHamiltonian, p, q):
    size = 3 * q
    """H_qxq = [][]
    for _ in range(size):
        for _ in range(size):
            H_qxq[][] = """
    alpha = p / q
    H_mageneticCell = np.zeros([size, size], dtype=complex)

    for i in range(size):
        for j in range(size):
            matrix_ith = 0
            for matrix in listHamiltonian:

                if i == j:
                    H_mageneticCell[i][j] = matrix[0][0]
                elif i == j + 1:
                    H_mageneticCell[i][j] = -t1 * np.exp(2j * np.pi * alpha * matrix_ith)
                elif i == j - 1:
                    H_mageneticCell[i][j] = -t1 * np.exp(2j * np.pi * alpha * matrix_ith)
                elif i == j + 3:
                    H_mageneticCell[i][j] = -t1
                elif i == j - 3:
                    H_mageneticCell[i][j] = -t1

                matrix_ith += 1

    # print(H_mageneticCell, "\n")

    return H_mageneticCell


def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)


def hamiltonian(a, b, n):
    h0 = 2 * t0 * (cos(2 * a) + 2 * cos(n) * cos(a) * cos(b) + 2 * sin(n) * sin(a) * cos(b)) + e1
    h1 = 2j * t1 * (sin(2 * a) + cos(n) * sin(a) * cos(b) - sin(n) * cos(a) * cos(b)) - 2 * sqrt(3) * t2 * (
        cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
    )
    h2 = 2 * t2 * (cos(2 * a) - cos(n) * cos(a) * cos(b) - sin(n) * sin(a) * cos(b)) + 2j * sqrt(3) * t1 * (
        cos(n) * cos(a) * sin(b) + sin(n) * sin(a) * sin(b)
    )
    h11 = (t11 + 3 * t22) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t11 * cos(2 * a) + e2
    h22 = (3 * t11 + t22) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t22 * cos(2 * a) + e2
    h12 = 2j * t12 * (sin(a) * cos(a) - cos(n) * sin(a) * cos(b) + sin(n) * cos(a) * cos(b)) + sqrt(3) * (t22 - t11) * (
        cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
    )

    h1dag = complex(h1).conjugate()
    h2dag = complex(h2).conjugate()
    h12dag = complex(h12).conjugate()

    return [[h0, h1, h2], [h1dag, h11, h12], [h2dag, h12dag, h22]]


def eigenvalue(q_max):
    p_q_ratios = []
    eigenvalue_data = []
    H_list = []

    for q in tqdm(range(1, q_max + 1)):
        for p in range(1, q_max + 1):
            if gcd(p, q) == 1:
                y = np.zeros(3 * q)
                a = (2 * pi) / (3 * q)
                b = 0
                alpha = p / q
                n = 2 * pi * alpha
                y[:] = alpha
                H_list.append(hamiltonian(a, b, n))

                H_mageneticCell = tridiagonal_matrix(H_list, p, q)

                eigenvalues = np.linalg.eigvalsh(H_mageneticCell)

                for i in range(len(eigenvalues)):
                    plt.plot(eigenvalues[i], y[:1], "o", c="red", markersize=0.1)

                # print(eigenvalues)
    plt.show()


eigenvalue(20)
