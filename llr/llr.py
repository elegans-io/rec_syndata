import math
import numpy as np

# Created by Mario Alemi 29 November 2017
def xlogx(x):
    return -x * math.log(float(x)) if x > 0 else 0


def xlogx_vec(x):
    xlogx_vectorized = np.vectorize(xlogx)
    return np.array([xlogx_vectorized(a) for a in x])


def entropy2(a, b):
    return -xlogx(a + b) + xlogx(a) + xlogx(b)


def entropy4(a, b, c, d):
    return -xlogx(a + b + c + d) + xlogx(a) + xlogx(b) + xlogx(c) + xlogx(d)


def loglikelihood_ratio(k11, k10, k01, k00):
    assert (k11 >= 0 and k10 >= 0 and k01 >= 0 and k00 >= 0)
    row_entropy = entropy2(k11 + k10, k01 + k00)
    column_entropy = entropy2(k11 + k01, k10 + k00)
    matrix_entropy = entropy4(k11, k10, k01, k00)
    if row_entropy + column_entropy < matrix_entropy:
        # round off error
        return 0.0
    return 2.0 * (row_entropy + column_entropy - matrix_entropy)


def root_loglikelihood_ratio(k11, k10, k01, k00):
    result = k11.copy()
    result.fill(0.0)
    for i in range(len(result)-1):
        for j in range(len(result[i])-1):
            result[i][j] = np.sqrt(loglikelihood_ratio(k11[i][j], k10[i][j], k01[i][j], k00[i][j]))
            if (k11[i][j] + k10[i][j]) > 0 and (k01[i][j] + k00[i][j]) > 0:
                if k11[i][j] / (k11[i][j] + k10[i][j]) < k01[i][j] / (k01[i][j] + k00[i][j]):
                    result[i][j] = -result[i][j]
    return result
