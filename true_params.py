from macros import *
from lds import LDS

m, n, p = 2, 2, 2
LDS_LIST = [
    LDS(A = np.array([[1, 0], [0, 1]]), B = np.array([[1, 0], [0, 1]]), C = np.array([[1, 0], [0, 1]]), D = np.array([[1, 0], [0, 1]])),
    LDS(A = np.array([[0, 1], [1, 0]]), B = np.array([[1, 0], [0, 1]]), C = np.array([[1, 0], [0, 0]]), D = np.array([[1, 0], [0, 1]])),
]
k = len(LDS_LIST)

def generate_samples(lds_list, weights, s, num_trajectories = 100):
    length = 4*s + 2
    sample_from = np.random.choice(lds_list, size=num_trajectories, p=weights)
    trajectories = [lds.generate_trajectory(length) for lds in sample_from]
    return trajectories

def get_expected_Pi_M(lds_list, weights, s):
    flattened_markov_params = [np.concatenate((lds.D.flatten(),) + tuple((lds.C @ mpow(lds.A, i) @ lds.B).flatten() for i in range(2*s))) for lds in lds_list]
    indiv_Pi_M = [np.outer(vec, vec) for vec in flattened_markov_params]
    return np.average(indiv_Pi_M, axis=0, weights=weights)

def get_expected_R(lds_list, weights, s):
    return np.average([[lds.D if i == 0 else lds.C @ mpow(lds.A, i-1) @ lds.B for i in range(4*s+2)] for lds in lds_list], axis=0, weights=weights)