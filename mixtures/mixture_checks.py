from macros import *

def get_indiv_Pi_M(lds_list, s):
    flattened_markov_params = [np.concatenate((lds.D.flatten(),) + tuple((lds.C @ mpow(lds.A, i) @ lds.B).flatten() for i in range(2*s))) for lds in lds_list]
    return [np.outer(vec, vec) for vec in flattened_markov_params]

def get_expected_Pi_M(lds_list, weights, s):
    return np.average(get_indiv_Pi_M(lds_list, s), axis=0, weights=weights)

def get_expected_random_weighted_Pi_M(lds_list, weights, s, differs_at, random_weighting):
    random_weights = np.array([np.dot(lds.get_markov_param(differs_at).flatten(), random_weighting) for lds in lds_list])
    return np.average(get_indiv_Pi_M(lds_list, s), axis=0, weights=random_weights * weights)

def get_expected_R(lds_list, weights, s):
    return np.average([[lds.get_markov_param(i) for i in range(2*s+1)] for lds in lds_list], axis=0, weights=weights).flatten()