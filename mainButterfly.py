"""n = 20  # Tương ứng 3n
cols = 3 * n  # Số cột
rows = 3 * n  # Số hàng"""

import numpy as np
from numpy import cos, sin, pi, sqrt, conjugate
from math import exp
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=10, linewidth=1200, edgeitems=24)
# Các hằng số
e1 = 1.046
e2 = 2.104
t0 = -0.184
t1 = 0.401
t2 = 0.507
t11 = 0.218
t12 = 0.338
t22 = 0.057


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def generatorHamiltonian(p, q, n):
    alpha = p / q
    a = (2 * pi) / 3 * alpha
    b = 0 * alpha
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

    h1dag = conjugate(h1)
    h2dag = conjugate(h2)
    h12dag = conjugate(h12)

    H = np.array([[h0, h1, h2], [h1dag, h11, h12], [h2dag, h12dag, h22]])

    return H


def reWriteHamiltonian(p, q, listHamiltonian):
    alpha = p / q
    a = (2 * pi) / 3 * alpha

    magneticHamiltonian = []


def eigenvalue(q_max):
    listHamiltonian = []
    for p in tqdm(range(1, q_max + 1)):
        for q in range(1, q_max + 1):
            if q > p:
                if gcd(p, q) == 1:
                    eta = p / q
                    H = generatorHamiltonian(p, q, eta)
                    # print(H, f"{p}/{q}", "\n")
                    listHamiltonian.append(H)

                    listHamiltonian.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
                    listHamiltonian.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    print(np.array(listHamiltonian))


def main():
    eigenvalue(q_max=3)


if __name__ == "__main__":
    main()
