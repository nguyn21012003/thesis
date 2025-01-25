import numpy as np
from numpy import cos, sin, sqrt
from math import pi
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=10, linewidth=120, edgeitems=24)

q = 51  # Size of lattice along x direction

alpharr = np.zeros(3 * q)  ### Chứa tất cả các thành phần của p / q

for i in range(1, q):
    alpharr[i] = i / q  ### lấy p / q. Trong đó q là cố định i sẽ là p.

print(alpharr)


nu_arr = np.linspace(0, 2 * np.pi, 3 * q)
print(nu_arr)
e1 = 1.046
e2 = 2.104
t0 = -0.184
t1 = 0.401
t2 = 0.507
t11 = 0.218
t12 = 0.338
t22 = 0.057


# Periodic boundary conditions
def pbc(i):
    if i == q:
        return 0
    elif i == -1:
        return q - 1
    else:
        return i


def Hamiltonian(alpha, nu):
    H = np.zeros([q, q])
    for i in range(1, q):
        a = 4 * pi / (3)
        b = 0

        # H[i, i] = 2 * t0 * (cos(2 * a * i) + 2 * cos(nu) * cos(a * i) * cos(b * i) + 2 * sin(nu) * sin(a * i) * cos(b * i)) + e1
        H[i, i] = 2 * np.cos((2 * pi * i * alpha) - nu)
        H[i, pbc(i + 1)] = 1.0
        H[i, pbc(i - 1)] = 1.0
    # print(H, "\n")
    return H


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

    h1dag = np.conjugate(h1)
    h2dag = np.conjugate(h2)
    h12dag = np.conjugate(h12)

    H = [[h0, h1, h2], [h1dag, h11, h12], [h2dag, h12dag, h22]]
    H = np.array(H)

    return H


def tridiagonal_matrix(H_list, I, q):
    size = 3 * q
    """H_qxq = [][]
    for _ in range(size):
        for _ in range(size):
            H_qxq[][] = """

    H_qxq = [[0 for _ in range(size)] for _ in range(size)]

    for k in range(q):
        start_row = 3 * k
        start_col = 3 * k

        for i in range(3):
            for j in range(3):
                H_qxq[start_row + i][start_col + j] = H_list[k][i][j]

        if start_col + 3 < size:
            for i in range(3):
                for j in range(3):
                    H_qxq[start_row + i][start_col + 3 + j] = I[i][j]

        if start_row + 3 < size:
            for i in range(3):
                for j in range(3):
                    H_qxq[start_row + 3 + i][start_col + j] = I[i][j]

    return H_qxq


y_vals = np.zeros(3 * q)
x_vals = np.zeros(3 * q)
for alpha in tqdm(alpharr):
    H_list = []
    y_vals[:] = alpha
    for nu in nu_arr:  # Diagonalze Hamiltonaian. Plot energies for each alpha ..(\eps,\alpha)
        a = a = 4 * pi * alpha / 6
        b = 0
        hamilton = hamiltonian(a, b, nu)
        H_list.append(hamilton)
        # print(H_list, "\n")

        # x_vals = np.linalg.eigvalsh(hamilton)
        #

H = tridiagonal_matrix(H_list, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], q)
x_vals = np.linalg.eigvalsh(H)
plt.plot(x_vals, y_vals, "o", markersize=0.2)
plt.show()
