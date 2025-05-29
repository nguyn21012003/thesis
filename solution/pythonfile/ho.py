import csv
import sys

import numpy as np
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm

from datetime import datetime

from file_python.condition import pbc
from file_python.HamTest import HamTNN_test
from file_python.HamTMD import Hamiltonian
from file_python.HamTMDNN import HamTNN
from file_python.HamTri import HamTriangular
from file_python.HamTriNN import HamTriNN
from file_python.irrMatrix import IR, IRNN, IRTNN
from file_python.parameters import paraNN, paraTNN
from file_python.plotbyGNU import PlotMatrixGNU
from file_python.squareHam import H as HamSquare
from file_python.ultilities import saveFunction
from tes import HamiltonianTest


def saveMatrix(matrix, fileName):
    with open(fileName, "w", newline="") as Matrixfile:
        header = ["row", "column", "value"]
        writer = csv.DictWriter(Matrixfile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                Matrixfile.write(f"{i + 1} {j + 1} {np.abs(matrix[i, j])}\n")
            Matrixfile.write("\n")

    return None


def process(band, choice: int, n_levels, qmax, kpoint, file, fileMatrix, file_plot_Matrix_Gnu, fileLandauLevels_valenceBand, fileLandauLevels_conductionBand, filegnu):
    model = "LDA"
    # matt, alattice, e1, e2, t0, t1, t2, t11, t12, t22 = paraNN(choice)
    data = paraTNN(choice, model)
    matt = data["material"]
    alattice = data["alattice"]

    E0, h1, h2, h3, h4, h5, h6 = IR(data)
    o1, o2, o3, o4, o5, o6 = IRTNN(data)
    v1, v2, v3, v4, v5, v6 = IRNN(data)

    irreducibleMatrix = {
        "NN": [E0, h1, h2, h3, h4, h5, h6],
        "TNN": [o1, o2, o3, o4, o5, o6],
        "NNN": [v1, v2, v3, v4, v5, v6],
    }

    kpoints = {
        "G": [0, 0],
        "K": [4 * pi / (3 * alattice), 0],
        "M": [pi / (alattice), pi / (sqrt(3) * alattice)],
    }

    kx = kpoints[kpoint][0]
    ky = kpoints[kpoint][1]

    with open(file, "w", newline="") as writefile:
        header = [
            "eta",
            "Energy",
            f"p/{qmax}",
        ]

        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        for p in tqdm(range(1, qmax + 1), ascii=" #", desc=f"{matt}"):
            if np.gcd(p, qmax) != 1:
                continue
            eta = p / (qmax)
            # Ham = HamTriNN(band, choice, alattice, p, qmax, pbc, kx, ky)
            Ham = HamTNN(band, alattice, p, 1 * qmax, kx, ky, irreducibleMatrix)
            # Ham = Hamiltonian(band, choice, alattice, p, qmax, pbc, kx, ky, E_R0, E_R1, E_R2, E_R3, E_R4, E_R5, E_R6, E_R7, E_R8, E_R9, E_R10, E_R11, E_R12)

            # funcName = Ham.__name__
            eigenvalue1 = LA.eigvalsh(Ham)
            for i in range(len(eigenvalue1)):
                writer.writerow(
                    {
                        "eta": eta,
                        "Energy": eigenvalue1[i],
                        f"p/{qmax}": f"{p}/{qmax}",
                        # "l1": eigenvalue2[0][i],:writeheader
                        # "l2": eigenvalue2[1][i],
                        # "l3": eigenvalue2[2][i],
                    }
                )
                writefile.write("\n")
            # saveMatrix(Ham, fileMatrix)
            # PlotMatrixGNU(fileMatrix, file_plot_Matrix_Gnu)
            # plotMatrix(H)

    return None


def main():
    qmax = 199
    n_levels = 8
    choice = 0
    band = 1
    model = "GGA"
    kpoint = "G"
    data = paraTNN(choice, model)
    matt = data["material"]
    time_run = datetime.now().strftime("%a-%m-%Y")

    currentProg = sys.argv[0].replace(".py", "")
    file = f"{band}band_dataHofstadterButterfly_q_{qmax}{time_run}_down.dat"

    fileMatrix = f"{band}band_Matrix_q_{qmax}{time_run}.dat"
    file_plot_Matrix_Gnu = f"{band}band_Matrix_q_{qmax}{time_run}_h0.gnuplot"

    fileLandauLevels_conductionBand = f"LandauLevels_q={qmax}_{matt}_c_band.dat"
    fileLandauLevels_valenceBand = f"LandauLevels_q={qmax}_{matt}_v2_band.dat"
    fileLandauLevels_1 = f"LandauLevels_q={qmax}_{matt}_1_band.dat"
    fileLandauLevels_2 = f"LandauLevels_q={qmax}_{matt}_2_band.dat"
    fileLandauLevels_3 = f"LandauLevels_q={qmax}_{matt}_3_band.dat"

    print(fileLandauLevels_1)
    print(fileLandauLevels_2)
    print(fileLandauLevels_3)

    filegnu = f"{band}band_plotHofstadterButterfly_q={qmax}.gnuplot"

    print("file data: ", file)
    print("file gnuplot: ", filegnu)
    print("file landau: ", fileLandauLevels_conductionBand, "&", fileLandauLevels_valenceBand)
    print("file Matrix: ", fileMatrix)
    print("file Matrix GNU: ", file_plot_Matrix_Gnu)
    # LandauLevels(fileLandauLevels_conductionBand, fileLandauLevels_valenceBand, fileLandauLevels_1, fileLandauLevels_2, fileLandauLevels_3, qmax, n_levels, choice)
    data = process(band, choice, n_levels, qmax, kpoint, file, fileMatrix, file_plot_Matrix_Gnu, fileLandauLevels_valenceBand, fileLandauLevels_conductionBand, filegnu)

    # saveFunction(currentProg, "fileData")

    return None


if __name__ == "__main__":
    main()
