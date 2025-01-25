import numpy as np
import matplotlib.pyplot as plt
from math import gcd, pi, cos, sin, sqrt
from tqdm import tqdm

# Định nghĩa kích thước q_max
q_max = 5


# Hàm tạo Hamiltonian 3x3
def create_H(a, b, n, t0, t1, t2, t11, t22, t12, e1, e2):
    h0 = 2 * t0 * (cos(2 * a) + 2 * cos(n) * cos(a) * cos(b) + 2 * sin(n) * sin(a) * cos(b)) + e1
    h1 = 2j * t1 * (sin(2 * a) + cos(n) * sin(a) * cos(b) - sin(n) * cos(a) * cos(b)) - 2 * sqrt(3) * t2 * (
        cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
    )
    h2 = 2 * t2 * (cos(2 * a) - cos(n) * cos(a) * cos(b) - sin(n) * sin(a) * cos(b)) + 2j * sqrt(3) * t1 * (
        cos(n) * cos(a) * sin(b) + sin(n) * sin(a) * sin(b)
    )
    h11 = (t11 + 3 * t22) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t11 * cos(2 * a) + e2
    h22 = (t22 + 3 * t11) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t22 * cos(2 * a) + e2
    h12 = 2j * t12 * (sin(a) * cos(a) - cos(n) * sin(a) * cos(b) + sin(n) * cos(a) * cos(b)) + sqrt(3) * (t22 - t11) * (
        cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
    )
    h1dag = np.conjugate(h1)
    h2dag = np.conjugate(h2)
    h12dag = np.conjugate(h12)

    return np.array([[h0, h1, h2], [h1dag, h11, h12], [h2dag, h12dag, h22]])


# Khởi tạo các tham số cố định
a, b = 1.0, 1.0
t0, t1, t2, t11, t22, t12 = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
e1, e2 = 1.0, 1.0

# Tìm các giá trị alpha hợp lệ
alpha_values = [(p, q) for p in range(1, q_max + 1) for q in range(1, q_max + 1) if q > p and gcd(p, q) == 1]
alpha_count = len(alpha_values)

# Tạo ma trận lớn Toeplitz
big_matrix_size = 3 * alpha_count
big_matrix = np.zeros((big_matrix_size, big_matrix_size), dtype=complex)
identity_matrix = np.eye(3, dtype=complex)  # Ma trận đơn vị 3x3

# Duyệt qua các alpha và điền ma trận lớn
index = 0
for p, q in tqdm(alpha_values):
    alpha = p / q
    n_values = np.linspace(0, 2 * pi, q, endpoint=False)  # Giá trị n chạy từ 0 đến 2π/q
    for n in n_values:
        H = create_H(a, b, n, t0, t1, t2, t11, t22, t12, e1, e2)

        # Điền H vào đường chéo
        big_matrix[index * 3 : (index + 1) * 3, index * 3 : (index + 1) * 3] = H

        # Điền ma trận I vào các vị trí gần kề (kiểu Toeplitz)
        if index > 0:
            big_matrix[index * 3 : (index + 1) * 3, (index - 1) * 3 : index * 3] = identity_matrix
            big_matrix[(index - 1) * 3 : index * 3, index * 3 : (index + 1) * 3] = identity_matrix
        index += 1

# Tính giá trị riêng (eigenvalues)
eigenvalues = np.linalg.eigvalsh(big_matrix)

# Vẽ đồ thị Hofstadter Butterfly
plt.figure(figsize=(10, 8))
for i, (p, q) in enumerate(alpha_values):
    alpha = p / q
    plt.scatter([alpha] * len(eigenvalues), eigenvalues, color="blue", s=0.5)

# Gán nhãn trục và hiển thị
plt.xlabel(r"$\alpha = p/q$", fontsize=15)
plt.ylabel(r"$\epsilon$", fontsize=15)
plt.title("Hofstadter Butterfly with Toeplitz Structure (3q x 3q)", fontsize=18)
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()
