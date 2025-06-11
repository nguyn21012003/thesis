import csv
from datetime import datetime

import numpy as np
from numpy import linalg as LA
from numpy import pi, sqrt
from tqdm import tqdm

from file_python.HamTMD import Hamiltonian as HamNN
from file_python.HamTMDNN import HamTNN
from file_python.HamTMDNN_kx import HamTNN_kx
from file_python.HamTMDNN_ky import HamTNN_ky
from file_python.irrMatrix import IR, IRNN, IRTNN
from file_python.parameters import paraNN, paraTNN


def process(N: int, band: int, choice: int, qmax: int, kpoint: str, fileData: dict, model: str):

    fileEnergy = fileData["fileEnergy"]
    fileMoment = fileData["fileMoment"]

    data = paraTNN(choice, model)
    a_lattice = data["alattice"]
    E0, h1, h2, h3, h4, h5, h6 = IR(data)
    v1, v2, v3, v4, v5, v6 = IRNN(data)
    o1, o2, o3, o4, o5, o6 = IRTNN(data)
    m0 = 5.6770736 / 100
    hb = 0.658229
    irreducibleMatrix = {
        "NN": [E0, h1, h2, h3, h4, h5, h6],
        "NNN": [v1, v2, v3, v4, v5, v6],
        "TNN": [o1, o2, o3, o4, o5, o6],
    }

    p = 1

    PxArr = np.zeros((N, N))
    PyArr = np.zeros((N, N))
    pPlusArr = np.zeros((N, N))
    pMinusArr = np.zeros((N, N))
    moduloPArr = np.zeros((N, N))
    # dHam_kx = np.zeros((6 * qmax, 6 * qmax), dtype=complex)
    # dHam_ky = np.zeros((6 * qmax, 6 * qmax), dtype=complex)

    arrEigen = {}
    for q in tqdm(range(6 * qmax), desc="Create array eigenvalue"):
        arrEigen[f"L_{q+1}"] = np.zeros([N, N])

    akx, aky = np.zeros((N, N)), np.zeros((N, N))
    dk = (4 * pi / a_lattice) / (N - 1)
    for i1 in range(N):
        for j1 in range(N):
            akx[i1][j1] = (-2 * pi / a_lattice + (i1) * dk) * 1
            aky[i1][j1] = (-2 * pi / a_lattice + (j1) * dk) * 1

    for i in tqdm(range(N)):
        for j in tqdm(range(N), desc="vong lap j"):
            if np.gcd(p, qmax) != 1:
                continue

            Ham = HamTNN(band, a_lattice, p, 2 * qmax, akx[i][j], aky[i][j], irreducibleMatrix)
            dHam_kx = HamTNN_kx(band, a_lattice, p, 2 * qmax, akx[i][j], aky[i][j], irreducibleMatrix)
            dHam_ky = HamTNN_ky(band, a_lattice, p, 2 * qmax, akx[i][j], aky[i][j], irreducibleMatrix)

            eigenvalue, eigenvector = LA.eigh(Ham)

            for q in range(6 * qmax):
                arrEigen[f"L_{q+1}"] = eigenvalue[q]

            sumpx = 0 + 0j
            sumpy = 0 + 0j
            for bandi in range(6 * qmax):
                for bandj in range(6 * qmax):

                    sumpx += np.conjugate(eigenvector[qmax][bandj]) * dHam_kx[bandi][bandj] * eigenvector[qmax + 1][bandi]
                    sumpy += np.conjugate(eigenvector[qmax][bandj]) * dHam_ky[bandi][bandj] * eigenvector[qmax + 1][bandi]
            px = sumpx * m0 / hb
            py = sumpy * m0 / hb
            moduloP = sqrt(abs(px) ** 2 + abs(py) ** 2)
            pPlus = px + 1j * py
            pMinus = px - 1j * py

            PxArr = px
            PyArr = py
            moduloPArr[i][j] = moduloP
            pPlusArr[i][j] = abs(pPlus)
            pMinusArr[i][j] = abs(pMinus)

    with open(fileEnergy, "w", newline="") as writefile:
        header = [
            "kx",
            "ky",
        ]
        for q in range(6 * qmax):
            header.append(list(arrEigen.keys())[q])

        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        row = {}
        for i in range(N):
            for j in range(N):
                row["kx"] = akx[i][j] / (2 * pi / a_lattice)
                row["ky"] = aky[i][j] / (2 * pi / a_lattice)

                for k in range(6 * qmax):
                    row[f"L_{k+1}"] = arrEigen[f"L_{k+1}"]

                writer.writerow(row)
            writefile.write("\n")

    with open(fileMoment, "w", newline="") as writefile:
        print(pPlusArr)
        print(len(pPlusArr))
        header = [
            "kx",
            "ky",
            "pPlus",
            "pMinus",
            "pAbs",
        ]

        writer = csv.DictWriter(writefile, fieldnames=header, delimiter=",")
        writer.writeheader()
        row = {}
        for i in range(N):
            for j in range(N):
                row["kx"] = akx[i][j] / (2 * pi / a_lattice)
                row["ky"] = aky[i][j] / (2 * pi / a_lattice)
                row["pPlus"] = pPlusArr[i][j]
                row["pMinus"] = pMinusArr[i][j]
                row["pAbs"] = moduloPArr[i][j]

                writer.writerow(row)
            writefile.write("\n")

    return None


def main():
    qmax = 100
    choice = 0
    band = 3
    N = 49
    model = "GGA"
    kpoint = "M"
    data = paraTNN(choice, model)
    matt = data["material"]
    time_run = datetime.now().strftime("%a-%m-%Y")
    fileEnergy = f"{band}band_Energy_q_{qmax}_{matt}_{time_run}_{model}.csv"
    fileMoment = f"{band}band_Momentum_q_{qmax}_{matt}_{time_run}_{model}.csv"

    filegnu = f"{band}band_plotHofstadterButterfly_q={qmax}.gnuplot"

    fileData = {"fileEnergy": fileEnergy, "fileMoment": fileMoment}
    data = process(N, band, choice, qmax, kpoint, fileData, model)

    # PlotMatrixGNU(fileMatrix, file_plot_Matrix_Gnu)
    # saveFunction(currentProg, "fileData")

    return None


if __name__ == "__main__":
    main()
