from macros import *
from lds import LDS
from tqdm import tqdm

def get_markov_params_cov(trajectories, s):
    return [np.mean([
                     np.outer(y[i+k1], u[i]) for u, y in trajectories for i in range(len(u) - k1)
            ], axis=0) for k1 in range(2*s+1)]

def get_markov_params_regression(trajectories, s):
    traj_len, p = trajectories[0][0].shape
    coeffs = np.empty((len(trajectories)*traj_len, (2*s+1)*p))
    targets = np.concatenate([y for _, y in trajectories], axis=0)
    for i, (u, _) in enumerate(trajectories):
        flat_inp = np.concatenate([np.zeros(2*s*p), np.concatenate(u)])
        for j in range(traj_len):
            coeffs[i*traj_len + j] = flat_inp[j*p:(j+2*s+1)*p]
    
    markov_params = la.lstsq(coeffs, targets)[0].reshape((2*s+1, p, -1)).transpose((0, 2, 1))
    return np.flip(markov_params, axis=0)

def get_lds_parameters(markov_params, m, n, p):
    markov_params = np.array(markov_params).reshape((-1, m, p))
    D = markov_params[0]
    s = len(markov_params)//2
    
    # Ho-Kalman
    H = np.block([[markov_params[i+j+1] for j in range(s+1)] for i in range(s)])
    H_minus, H_plus = H[:, :p*s], H[:, -p*s:]
    U, S, Vh = la.svd(H_minus)
    U, S, Vh = U[:,:n], S[:n], Vh[:n,:]
    O, Q = U * np.sqrt(S), np.sqrt(S.reshape((-1, 1))) * Vh
    C, B = O[:m], Q[:,:p]
    A = la.pinv(O) @ H_plus @ la.pinv(Q)
    return A, B, C, D

def learn_single_cov(trajectories, s, m, n, p):
    return LDS(*get_lds_parameters(get_markov_params_cov(trajectories, s), m, n, p))

def learn_single_regression(trajectories, s, m, n, p):
    return LDS(*get_lds_parameters(get_markov_params_regression(trajectories, s), m, n, p))