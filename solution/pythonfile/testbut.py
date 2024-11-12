# Numpy modules for linear algebra calculations
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=10, linewidth=120, edgeitems=12)


# Define function of Harper-Hofstadter Hamiltonian for a three-band model
def H_three_band(p, q, kx, ky):
    # Define magnetic flux per unit cell
    alpha = p / q

    # Initialize a 3q x 3q zero matrix for the three-band model
    M = np.zeros([3 * q, 3 * q], dtype=complex)

    # Matrix elements for each band (1, 2, 3)
    for i in range(0, q):
        for band in range(3):
            idx = i + band * q  # Calculate the correct index in the matrix

            # Diagonal elements for the three bands
            M[idx, idx] = 2 * np.cos(ky - 2 * np.pi * alpha * i) + band  # Offset diagonal for each band
            M[idx, idx + 1] = 2 * np.cos(ky - 2 * np.pi * alpha * i) + band  # Offset diagonal for each band
            M[idx, idx + 2] = 2 * np.cos(ky - 2 * np.pi * alpha * i) + band  # Offset diagonal for each band

            # Off-diagonal hopping elements within each band
            if i == q - 1:
                M[idx, idx - 1] = 1
            elif i == 0:
                M[idx, idx + 1] = 1
            else:
                M[idx, idx - 1] = 1
                M[idx, idx + 1] = 1

    # Bloch boundary conditions for three bands
    for band in range(3):
        idx1 = band * q
        idx2 = (band + 1) * q - 1
        if q == 2:
            M[idx1, idx2] = 1 + np.exp(-q * 1.0j * kx)
            M[idx2, idx1] = 1 + np.exp(q * 1.0j * kx)
        else:
            M[idx1, idx2] = np.exp(-q * 1.0j * kx)
            M[idx2, idx1] = np.exp(q * 1.0j * kx)
    print(M)

    return M


# A gcd function to set irrational values for alpha
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


# Plot function for three-band model
def plot_butterfly_three_band(q_max):
    # Iterations over alpha values
    for p in tqdm(range(1, q_max + 1)):
        for q in range(1, q_max + 1):
            # Set alpha rational values less than 1
            if q > p and gcd(p, q) == 1:
                # Define alpha
                alpha = p / q
                y = np.zeros(3 * q)
                y[:] = alpha

                # Eigenvalues of the three-band Harper-Hofstadter matrix for each k value
                x1 = np.linalg.eigvalsh(H_three_band(p, q, kx=0, ky=0))
                x2 = np.linalg.eigvalsh(H_three_band(p, q, kx=np.pi / q, ky=np.pi / q))

                # Plot eigenvalues
                for i in range(len(x1)):
                    plt.plot([x1[i], x2[i]], y[:2], "-", c="black", markersize=0.1)

                plt.plot(x1, y, "o", c="black", markersize=0.1)
                plt.plot(x2, y, "o", c="black", markersize=0.1)

    plt.xlabel(r"$\epsilon$", fontsize=15)
    plt.ylabel(r"$\alpha$", fontsize=15)
    plt.title(r"$q=1-$" + str(q_max) + " for Three-Band Model")
    plt.show()


# Maximum q value
q_max = 10

# Plot the butterfly for three-band model
plot_butterfly_three_band(q_max)
