import numpy as np
from numpy import cos, sin, pi, sqrt
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the constants: MoS2
e1 = 1.046
e2 = 2.104
t0 = -0.184
t1 = 0.401
t2 = 0.507
t11 = 0.218
t12 = 0.338
t22 = 0.057
a_lattice = 0.3190  # nm
hbar = 6.582
B = 12
eB = 10e-3 * B
mass = 5.68
N = 100  # Grid size for k-points


def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


# Function to build H_qxq with 3 diagonal structure
def build_tridiagonal_matrix(H_list, t1, t2, q):
    # Initialize a 3q x 3q zero matrix
    H_qxq = np.zeros((3 * q, 3 * q), dtype=complex)

    # Main diagonal
    for i in range(q):
        H_qxq[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = H_list[i]

        # First upper and lower diagonals with t1
        if i < q - 1:
            H_qxq[3 * i : 3 * (i + 1), 3 * (i + 1) : 3 * (i + 2)] = np.eye(3)  # Upper diagonal (t1 as identity matrix)
            H_qxq[3 * (i + 1) : 3 * (i + 2), 3 * i : 3 * (i + 1)] = np.eye(3)  # Lower diagonal (t1 as identity matrix)

        # Second upper and lower diagonals with t2
        if i < q - 2:
            H_qxq[3 * i : 3 * (i + 1), 3 * (i + 2) : 3 * (i + 3)] = np.eye(3)  # Upper diagonal (t2 as identity matrix)
            H_qxq[3 * (i + 2) : 3 * (i + 3), 3 * i : 3 * (i + 1)] = np.eye(3)  # Lower diagonal (t2 as identity matrix)

    return H_qxq


# Function to calculate eigenvalues for a given q and p
def eigenvalue(q_max):
    p_q_ratios = []
    eigenvalue_data = []

    for p in tqdm(range(1, q_max + 1)):
        for q in range(1, q_max + 1):
            # Initialize k-points arrays
            lamda = []

            a = (2 * pi) / (3 * q)
            b = 0

            # Construct Hamiltonian for each m
            H_list = []
            if q > p:
                if gcd(p, q) == 1:
                    for m in range(1, q + 1):
                        n = 2 * pi * (p / q) * m

                        # Elements of the 3x3 Hamiltonian
                        h0 = 2 * t0 * (cos(2 * a) + 2 * cos(n) * cos(a) * cos(b) + 2 * sin(n) * sin(a) * cos(b)) + e1
                        h1 = 2j * t1 * (sin(2 * a) + cos(n) * sin(a) * cos(b) - sin(n) * cos(a) * cos(b)) - 2 * sqrt(3) * t2 * (
                            cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
                        )
                        h2 = 2 * t2 * (cos(2 * a) - cos(n) * cos(a) * cos(b) - sin(n) * sin(a) * cos(b)) + 2j * sqrt(3) * t1 * (
                            cos(n) * cos(a) * sin(b) + sin(n) * sin(a) * sin(b)
                        )
                        h11 = (t11 + 3 * t22) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t11 * cos(2 * a) + e2
                        h22 = (3 * t11 + t22) * (cos(n) * cos(a) * cos(b) + sin(n) * sin(a) * cos(b)) + 2 * t22 * cos(2 * a) + e2
                        h12 = 4j * t12 * (sin(a) * cos(a) - cos(n) * sin(a) * cos(b) + sin(n) * cos(a) * cos(b)) + sqrt(3) * (t22 - t11) * (
                            cos(n) * sin(a) * sin(b) - sin(n) * cos(a) * sin(b)
                        )

                        h1dag = np.conjugate(h1)
                        h2dag = np.conjugate(h2)
                        h12dag = np.conjugate(h12)

                        # Hamiltonian 3x3 for each m
                        H = np.array([[h0, h1, h2], [h1dag, h11, h12], [h2dag, h12dag, h22]])
                        H_list.append(H)

                    # Use the new tridiagonal structure for H_qxq
                    H_qxq = build_tridiagonal_matrix(H_list, t1=np.eye(3), t2=np.eye(3), q=q)

                    # Calculate eigenvalues
                    eigenvalues = np.linalg.eigvalsh(H_qxq)

                    # Store p/q and eigenvalues for plotting
                    p_q_ratios.extend([p / q] * len(eigenvalues))
                    eigenvalue_data.extend(eigenvalues)
    print(H_qxq)

    # Convert to arrays for easy plotting
    p_q_ratios = np.array(p_q_ratios)
    eigenvalue_data = np.array(eigenvalue_data)

    # Plotting the results
    plt.figure(figsize=(8, 8))
    plt.scatter(p_q_ratios, eigenvalue_data, s=1, color="red")
    plt.title("Hofstadter Butterfly")
    plt.xlabel(r"$p/q$ (Flux ratio)")
    plt.ylabel("Energy (eV)")
    plt.grid(True)
    plt.show()


# Call the function with a maximum q value
eigenvalue(q_max=50)
