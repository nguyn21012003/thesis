"""n = 20  # Tương ứng 3n
cols = 3 * n  # Số cột
rows = 3 * n  # Số hàng"""

import numpy as np
from math import cos, sin, pi, sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm

# Các hằng số
e1 = 1.046
e2 = 2.104
t0 = -0.184
t1 = 0.401
t2 = 0.507
t11 = 0.218
t12 = 0.338
t22 = 0.057


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
                    H_qxq[start_row + i][start_col + 3 + j] = -I[i][j]

        if start_row + 3 < size:
            for i in range(3):
                for j in range(3):
                    H_qxq[start_row + 3 + i][start_col + j] = -I[i][j]

    return H_qxq


def gcd(a, b):
    return a if b == 0 else gcd(b, a % b)


def hamiltonian(a, b, n, p, q):
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
        for p in range(1, q + 1):
            if gcd(p, q) == 1:
                a = (2 * pi) / (3 * q)
                b = 0
                for m in range(1, q + 1):
                    n = 2 * pi * (p / q) * m
                    H_list.append(hamiltonian(a, b, n, p, q))

                H_qxq = tridiagonal_matrix(H_list, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], q)

                eigenvalues = []
                eigenvalues = np.linalg.eigvalsh(H_qxq)

                # Thu thập dữ liệu để vẽ
                p_q_ratios.extend([p / q] * len(eigenvalues))
                eigenvalue_data.extend(eigenvalues)
    print(len(H_list))

    # Vẽ đồ thị
    plt.figure(figsize=(8, 8))
    plt.plot(p_q_ratios, eigenvalue_data, "o", markersize=0.2, color="r")
    plt.title("Hofstadter Butterfly")
    plt.xlabel(r"$p/q$")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.show()


# Thực thi
eigenvalue(5)


"""# Khởi tạo ma trận array (3n x 3n) với giá trị 0
array = [[0 for _ in range(cols)] for _ in range(rows)]

# Chèn ma trận H và ma trận I vào array
for k in range(n):  # Duyệt qua các khối đường chéo
    start_row = 3 * k
    start_col_H = 3 * k
    start_col_I = 3 * k + 3  # Cột bắt đầu của ma trận I (bên trên H)
    start_row_I = 3 * k + 3  # Hàng bắt đầu của ma trận I(bên dưới H)
    # Chèn H vào phần đường chéo (start_row, start_col_H)
    for i in range(3):
        for j in range(3):
            array[start_row + i][start_col_H + j] = H[i][j]

    # Chèn I vào phần bên cạnh H (start_row, start_col_I)
    if start_col_I < cols:  # Kiểm tra xem có đủ không gian để chèn I
        for i in range(3):
            for j in range(3):
                array[start_row + i][start_col_I + j] = I[i][j]
                array[start_row_I + i][start_col_H + j] = I[i][j]

# In ma trận array
for row in array:
    print(' '.join(map(str, row)))
"""
