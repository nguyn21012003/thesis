def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


def pbc(i, q):
    # if i == -1:
    #     return q - 1
    # elif i == q:
    #     return 0
    # elif i == q + 1:
    #     return 1
    # elif i == q + 2:
    #     return 2
    # elif i == q + 3:
    #     return 3
    # elif i == -2:
    #     return q - 2
    # elif i == -3:
    #     return q - 3
    # else:
    #     return i
    #
    return i % (q)
