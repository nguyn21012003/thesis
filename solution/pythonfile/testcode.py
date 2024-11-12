import numpy as np
import matplotlib.pyplot as plt


def create_honeycomb_hamiltonian(N, phi):
    """
    Tạo ma trận Hamiltonian cho lưới tổ ong với N nguyên tử theo mỗi chiều.
    phi: tỷ lệ từ thông (tính theo đơn vị lượng tử từ thông).
    """
    num_sites = 2 * N * N  # Tổng số nguyên tử (2 nguyên tử A và B cho mỗi ô)

    # Ma trận Hamiltonian
    H = np.zeros((num_sites, num_sites), dtype=complex)

    # Vị trí của nguyên tử A và B
    for i in range(N):
        for j in range(N):
            site_A = 2 * (i * N + j)  # Chỉ số cho nguyên tử A
            site_B = 2 * (i * N + j) + 1  # Chỉ số cho nguyên tử B

            # Hopping giữa các nguyên tử gần nhất
            # Hàng xóm theo chiều dọc
            if i < N - 1:
                H[site_A, 2 * ((i + 1) * N + j)] = -1
                H[site_B, 2 * ((i + 1) * N + j)] = -1

            # Hàng xóm bên phải
            if j < N - 1:
                H[site_A, 2 * (i * N + (j + 1)) + 1] = -1
                H[site_B, 2 * (i * N + (j + 1))] = -1

            # Hàng xóm chéo: bên dưới bên trái
            if i < N - 1 and j > 0:
                H[site_A, 2 * ((i + 1) * N + (j - 1)) + 1] = -np.exp(1j * phi * i)
                H[site_B, 2 * ((i + 1) * N + (j - 1))] = -np.exp(-1j * phi * i)

            # Hàng xóm chéo: bên dưới bên phải
            if i < N - 1 and j < N - 1:
                H[site_A, 2 * ((i + 1) * N + (j + 1)) + 1] = -np.exp(-1j * phi * (i + 1))
                H[site_B, 2 * ((i + 1) * N + (j + 1))] = -np.exp(1j * phi * (i + 1))

    return H


def calculate_spectrum(N, num_phi):
    """
    Tính phổ năng lượng cho nhiều tỷ lệ từ thông khác nhau.
    N: số lượng nguyên tử theo mỗi chiều.
    num_phi: số lượng giá trị từ thông.
    """
    energies = []
    phis = np.linspace(0, 1, num_phi)

    for phi in phis:
        H = create_honeycomb_hamiltonian(N, 2 * np.pi * phi)
        eigvals = np.linalg.eigvalsh(H)  # Tính trị riêng
        energies.append(eigvals)

    return energies, phis


def plot_hofstadter_butterfly(N, num_phi):
    """
    Vẽ phổ năng lượng Hofstadter butterfly.
    """
    energies, phis = calculate_spectrum(N, num_phi)

    plt.figure(figsize=(12, 8))
    for i, eigs in enumerate(energies):
        plt.scatter(phis[i] * np.ones_like(eigs), eigs, color="black", s=0.5)

    plt.title("Hofstadter Butterfly on Honeycomb Lattice")
    plt.xlabel(r"Magnetic Flux (in units of $\Phi_0$)")
    plt.ylabel("Energy")
    plt.xlim(0, 1)
    plt.ylim(-3, 3)
    plt.grid()
    plt.show()


# Sử dụng hàm để vẽ butterfly
N = 10  # Kích thước lưới
num_phi = 300  # Số lượng giá trị từ thông
plot_hofstadter_butterfly(N, num_phi)
