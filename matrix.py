import sympy as sp
from HNN_nguyen import eigenvalue


def matrx(arg):
    return arg


def det(matrix):
    det = 0
    for i in range(len(matrix)):
        if i + 2 < len(matrix):
            a = matrix[i][0]
            b = matrix[i + 1][1]
            c = matrix[i + 2][2]
            d = matrix[i][0]
            e = matrix[i + 1][1]
            f = matrix[i + 2][2]
            g = matrix[i][2]
            h = matrix[i + 1][2]
            j = matrix[i + 2][2]
            det = a * (e * i - f * h) + b * (f * g - i * d) + c * (d * h - g * e)
    return det


def main():
    print(matrx(eigenvalue()))
    print(det(matrx(eigenvalue())))


if __name__ == "__main__":
    main()
