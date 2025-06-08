import numpy as np
import csv
from numpy import pi
from numpy import linalg as LA
from numpy import sqrt, cos, sin, exp
from tqdm import tqdm
from file_python.irrMatrix import IR, IRNN, IRTNN
from file_python.parameters import paraNN, paraTNN


np.set_printoptions(precision=10, linewidth=1200, edgeitems=20, suppress=True)


def eigenvalue(material: int, model: str, frameWork: str):
    N = 100
    data = paraTNN(material, model) if frameWork == "TNN" else paraNN(material, model)
    kpoint = "kx_ky"
    a_lattice = data["alattice"]
    E0, h1, h2, h3, h4, h5, h6 = IR(data)
    o1, o2, o3, o4, o5, o6 = IRTNN(data)
    v1, v2, v3, v4, v5, v6 = IRNN(data)
    eigenvalues = []
    L1 = np.zeros((N, N))
    L2 = np.zeros((N, N))
    L3 = np.zeros((N, N))
    dk = (4 * pi / a_lattice) / (N - 1)
    akx, aky = np.zeros((N, N)), np.zeros((N, N))
    for i1 in range(N):
        for j1 in range(N):
            akx[i1][j1] = (-2 * pi / a_lattice + (i1) * dk) * 1
            aky[i1][j1] = (-2 * pi / a_lattice + (j1) * dk) * 1
    for i in tqdm(range(N)):
        for j in range(N):
            alpha = akx[i][j] / 2 * a_lattice
            beta = sqrt(3) / 2 * aky[i][j] * a_lattice
            ham = (
                E0
                + exp(2j * alpha) * h1
                + exp(1j * (alpha - beta)) * h2
                + exp(1j * (-alpha - beta)) * h3
                + exp(-2j * alpha) * h4
                + exp(1j * (-alpha + beta)) * h5
                + exp(1j * (alpha + beta)) * h6
                + exp(4j * alpha) * o1
                + exp(2j * (alpha - beta)) * o2
                + exp(2j * (-alpha - beta)) * o3
                + exp(-4j * alpha) * o4
                + exp(2j * (-alpha + beta)) * o5
                + exp(2j * (alpha + beta)) * o6
                + exp(1j * (3 * alpha - beta)) * v1
                + exp(1j * (-2 * beta)) * v2
                + exp(1j * (-3 * alpha - beta)) * v3
                + exp(1j * (-3 * alpha + beta)) * v4
                + exp(1j * (2 * beta)) * v5
                + exp(1j * (3 * alpha + beta)) * v6
            )

            w = LA.eigvalsh(ham)
            L1[i, j] = float(w[0])
            L2[i, j] = float(w[1])
            L3[i, j] = float(w[2])

    with open(f"SmallMagHam_eigenvalue_{kpoint}_{N}.csv", "w", newline="") as writefile:
        header = [
            "kx",
            "ky",
            "Lambda1",
            "Lambda2",
            "Lambda3",
            # "Lambda4",
            # "Lambda5",
            # "Lambda6",
        ]
        writer = csv.DictWriter(writefile, fieldnames=header)
        writer.writeheader()
        for i in range(N):
            for j in range(N):
                writer.writerow(
                    {
                        "kx": akx[i, j] / (2 * pi / a_lattice),
                        "ky": aky[i, j] / (2 * pi / a_lattice),
                        "Lambda1": L1[i, j],
                        "Lambda2": L2[i, j],
                        "Lambda3": L3[i, j],
                        # "Lambda4": L4[i][j],
                        # "Lambda5": L5[i][j],
                        # "Lambda6": L6[i][j],
                    }
                )
            writefile.write("\n")


def main():
    material = 0
    model = "LDA"
    frameWork = "TNN"
    eigenvalue(material, model, frameWork)
    return None


if __name__ == "__main__":
    main()
