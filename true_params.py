from macros import *
from lds import LDS

m, n, p = 2, 2, 2
LDS_LIST = [
    LDS(A = np.array([[1, 0], [0, 1]]), B = np.array([[1, 0], [0, 1]]), C = np.array([[1, 0], [0, 1]]), D = np.array([[1, 0], [0, 1]])),
    LDS(A = np.array([[0, 1], [1, 0]]), B = np.array([[1, 0], [0, 1]]), C = np.array([[1, 0], [0, 0]]), D = np.array([[1, 0], [0, 1]])),
]
k = len(LDS_LIST)