import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=10, linewidth=120, edgeitems=24)


def H(p, q, kx, ky):

    alpha = p / q

    M = np.zeros([3 * q, 3 * q], dtype=complex)

    for i in range(0, q):
        M[i, i] = 2 * np.cos(ky - 2 * np.pi * alpha * i)
    for i in range(q, 2 * q):
        M[i, i] = 2 * np.cos(ky - 2 * np.pi * alpha * i)
    for i in range(2 * q, 3 * q):
        M[i, i] = 2 * np.cos(ky - 2 * np.pi * alpha * i)
        # M[i, i] = 2 * np.cos(ky - 2 * np.pi * alpha * i)
    for i in range(3 * q - 1):
        if i == q - 1:
            M[i, i - 1] = 1
        elif i == 0:
            M[i, i + 1] = 1
        else:
            M[i, i - 1] = 1
            M[i, i + 1] = 1

    # Bloch conditions
    if q == 2:
        M[0, q - 1] = 1 + np.exp(-q * 1j * kx)
        M[q - 1, 0] = 1 + np.exp(q * 1j * kx)
    else:
        M[0, q - 1] = np.exp(-q * 1j * kx)
        M[q - 1, 0] = np.exp(q * 1j * kx)

    # print(M, "\n")

    return M


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def plot_butterfly(q_max):

    for p in tqdm(range(1, q_max + 1)):
        for q in range(1, q_max + 1):
            if q > p:
                if gcd(p, q) == 1:

                    alpha = p / q
                    y = np.zeros(3 * q)
                    y[:] = alpha**q
                    # print(len(y))
                    # print(y)

                    x1 = np.linalg.eigvalsh(H(p, q, kx=0, ky=0))
                    x2 = np.linalg.eigvalsh(H(p, q, kx=np.pi / q, ky=np.pi / q))
                    # print(x2, len(x2), "\n")
                    # print(x1, len(x1), "\n")
                    for i in range(len(x1)):
                        plt.plot([x1[i], x2[i]], y[:2], "-", c="red", markersize=0.1)

                    plt.plot(x1, y, "o", c="orange", markersize=0.1)
                    plt.plot(x2, y, "o", c="blue", markersize=0.1)

    plt.xlabel(r"$\epsilon$", fontsize=15)
    plt.ylabel(r"$\alpha$", fontsize=15)
    plt.title(r"$q=1-$" + str(q_max))
    plt.show()


q = 30
plot_butterfly(q)
