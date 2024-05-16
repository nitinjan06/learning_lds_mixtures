from macros import *
from tqdm import tqdm
from lds import LDS
from learning_algos.learn_single import get_lds_parameters, learn_single_regression

def get_Pi_M_with_var(trajectories, s):
    Ts = [np.block([[np.outer(np.kron(u[0], y[k1]), np.kron(u[k1+1], y[k1 + k2 + 1])) for k1 in range(2*s+1)] for k2 in range(2*s+1)]) for u, y in trajectories]
    T_mean = np.mean(Ts, axis=0)
    T_var = np.var(Ts, axis=0)
    # doesn't symmetrize
    return T_mean, T_var
 
def get_equal_weighted_Pi_M(trajectories, s):
    ans = np.block([[
                       np.mean([
                           np.outer(np.kron(y[k1 + k2 + 1], u[k1+1]), np.kron(y[k1], u[0]))
                       for u, y in trajectories], axis=0)
                   for k1 in range(2*s + 1)] for k2 in tqdm(range(2*s + 1))])
    return (ans + ans.T)/2 # denoise

def get_random_weighting(m, p):
    return normal(size=m*p)

def get_random_weighted_Pi_M(trajectories, s, differs_at, random_weighting = None):
    if random_weighting is None:
        u0, y0 = trajectories[0]
        random_weighting = get_random_weighting(y0.shape[1], u0.shape[1])
    
    ans = np.block([[
                       np.mean([
                           np.outer(np.kron(y[differs_at + k1 + k2 + 2], u[differs_at + k1 + 2]), np.kron(y[differs_at + k1 + 1], u[differs_at + 1])) * np.dot(random_weighting, np.kron(y[differs_at], u[0]))
                       for u, y in trajectories], axis=0)
                   for k1 in range(2*s + 1)] for k2 in tqdm(range(2*s + 1))])
    return (ans + ans.T)/2

def extract_components(Pi_M1, Pi_M2, k):
    U1, S1, Vh1 = la.svd(Pi_M1, hermitian=True)
    U2, S2, Vh2 = la.svd(Pi_M2, hermitian=True)
    S1[k:] = 0
    S2[k:] = 0

    # because of symmetry, we actually don't need the full Jennrich's algorithm, we can get by with just the eigenvectors of U
    U = (U1 * S1) @ Vh1 @ (Vh2.T * np.reciprocal(S2, where=(S2 != 0))) @ U2.T
    #V = (U2 * S2) @ Vh2 @ (Vh1.T * np.reciprocal(S1, where=(S2 != 0))) @ U1.T
    U_lambdas, U_eigvecs = la.eig(U)
    U_lambdas, U_eigvecs = np.real(U_lambdas), np.real(U_eigvecs).T
    # V_lambdas, V_eigvecs = la.eig(V)
    return [vec for lmbda, vec in zip(U_lambdas, U_eigvecs) if np.abs(lmbda) > 1e-4]

def rescale_components(components, Pi_M):
    square_reweighting, _, _ , _ = la.lstsq(np.array([np.kron(vec, vec) for vec in components]).T, Pi_M.flatten())
    return components * np.sqrt(square_reweighting).reshape((-1, 1))

def get_R(trajectories, s):
    return np.array([np.mean([np.kron(y[k1], u[0]) for u, y in trajectories], axis=0) for k1 in range(2*s+1)]).flatten()

def get_weights_and_components(components, R):
    sqrt_weights, _, _, _ = la.lstsq(components.T, R)
    return sqrt_weights**2, components/sqrt_weights.reshape((-1, 1))

def unflatten_components(components, m, p):
    return components.reshape(components.shape[0], -1, m, p)

def label_trajectories(samples, lds_list):
    return np.argmin([[lds_list[i].get_nll(traj) for i in range(2)] for traj in samples], axis = 1)

def em_step(samples, labels, k, s, m, n, p):
    sorted_samples = [[] for _ in range(k)]
    for traj, label in zip(samples, labels):
        sorted_samples[label].append(traj)
    recovered = [learn_single_regression(sorted_samples[i], s, m, n, p) for i in range(k)]
    return recovered, label_trajectories(samples, recovered)


def learn_with_different_weights(samples_1, samples_2, k, s, m, n, p):
    Pi_Ms = [get_equal_weighted_Pi_M(samples_1, s), get_equal_weighted_Pi_M(samples_2, s)]
    pre_components = extract_components(Pi_Ms[0], Pi_Ms[1], k)
    components = rescale_components(pre_components, Pi_Ms[0])
    R = get_R(samples_1, s)
    weights, markov_param_list = get_weights_and_components(components, R)
    recovered_lds_list = [LDS(*get_lds_parameters(markov_params, m, n, p)) for markov_params in markov_param_list]
    return weights, recovered_lds_list

def learn_with_random_weighting(samples, k, s, m, n, p, random_weighting, differs_at = 2):
    Pi_Ms = [get_equal_weighted_Pi_M(samples, s), get_random_weighted_Pi_M(samples, s, differs_at, random_weighting)]
    pre_components = extract_components(Pi_Ms[0], Pi_Ms[1], k)
    components = rescale_components(pre_components, Pi_Ms[0])
    R = get_R(samples, s)
    weights, markov_param_list = get_weights_and_components(components, R)
    recovered_lds_list = [LDS(*get_lds_parameters(markov_params, m, n, p)) for markov_params in markov_param_list]
    return weights, recovered_lds_list

def get_random_labels(k, samples):
    return np.random.randint(k, size=len(samples))

def learn_with_em(samples, k, s, m, n, p, initial_labels = None):
    labels = initial_labels if initial_labels is not None else get_random_labels(k, samples)

    while True:
        try:
            EM_MAX_STEPS = 100
            CONVERGENCE_CUTOFF = 1
            for i in range(EM_MAX_STEPS):
                # print(f"on em step: {i+1}/{EM_MAX_STEPS}")
                lds_list, new_labels = em_step(samples, labels, k, s, m, n, p)
                similarity = (labels == new_labels).mean()
                labels = new_labels
                # print(f"similarity: {similarity}\n")
                if similarity >= CONVERGENCE_CUTOFF: break
            
            return new_labels, lds_list
        except:
            print("all labels were the same, retrying")
            labels = get_random_labels(k, samples)