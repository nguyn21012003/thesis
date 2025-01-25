import numpy as np

from math import sqrt
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import subprocess


np.set_printoptions(precision=10, linewidth=120, edgeitems=15)


def createMatrix(dim: int):

    AMatrix = np.zeros([dim**2, dim**2])

    for i in range(1, dim - 1):
        for j in range(1, dim - 1):
            north = (i - 1) * dim + j
            west = i * dim + j - 1
            index = i * dim + j
            east = i * dim + j + 1
            south = (i + 1) * dim + j

            AMatrix[index, north] = 1
            AMatrix[index, west] = 1
            AMatrix[index, index] = -4
            AMatrix[index, east] = 1
            AMatrix[index, south] = 1

    i = 0
    for j in range(1, dim - 1):
        west = i * dim + j - 1
        index = i * dim + j
        east = i * dim + j + 1
        south = (i + 1) * dim + j

        AMatrix[index, west] = 1
        AMatrix[index, index] = -4
        AMatrix[index, east] = 1
        AMatrix[index, south] = 1

    j = 0
    for i in range(1, dim - 1):
        north = (i - 1) * dim + j
        index = i * dim + j
        east = i * dim + j + 1
        south = (i + 1) * dim + j

        AMatrix[index, north] = 1
        # AMatrix[index,west] =1
        AMatrix[index, index] = -4
        AMatrix[index, east] = 1
        AMatrix[index, south] = 1

    j = dim - 1
    for i in range(1, dim - 1):
        north = (i - 1) * dim + j
        west = i * dim + j - 1
        index = i * dim + j
        # east = i*n+j+1
        south = (i + 1) * dim + j

        AMatrix[index, north] = 1
        AMatrix[index, west] = 1
        AMatrix[index, index] = -4
        # a[index,east] =1
        AMatrix[index, south] = 1

    i = dim - 1
    for j in range(1, dim - 1):
        north = (i - 1) * dim + j
        west = i * dim + j - 1
        index = i * dim + j
        east = i * dim + j + 1
        # south = (i+1)*n+j

        AMatrix[index, north] = 1
        AMatrix[index, west] = 1
        AMatrix[index, index] = -4
        AMatrix[index, east] = 1

    i = 0
    j = 0
    index = i * dim + j
    east = i * dim + j + 1
    south = (i + 1) * dim + j

    AMatrix[index, index] = -4
    AMatrix[index, east] = 1
    AMatrix[index, south] = 1

    i = 0
    j = dim - 1
    west = i * dim + j - 1
    index = i * dim + j
    south = (i + 1) * dim + j

    AMatrix[index, west] = 1
    AMatrix[index, index] = -4
    AMatrix[index, south] = 1

    i = dim - 1
    j = 0
    north = (i - 1) * dim + j
    index = i * dim + j
    east = i * dim + j + 1

    AMatrix[index, north] = 1
    AMatrix[index, index] = -4
    AMatrix[index, east] = 1

    i = dim - 1
    j = dim - 1
    north = (i - 1) * dim + j
    west = i * dim + j - 1
    index = i * dim + j

    AMatrix[index, north] = 1
    AMatrix[index, west] = 1
    AMatrix[index, index] = -4

    # print(AMatrix, "\n")

    return AMatrix


def main():
    numberLoop = 100
    ic = 0
    unknowPoints = 5
    i, j = unknowPoints, unknowPoints
    dim = unknowPoints

    AMatrix = createMatrix(dim)
    print(AMatrix)


if __name__ == "__main__":
    main()
