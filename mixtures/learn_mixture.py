from macros import *
from tqdm import tqdm

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
    
    # ans = np.block([[
    #                    np.mean([
    #                        np.outer(np.kron(y[k1 + k2 + 1], u[k1+1]), np.kron(y[k1], u[0])) * np.dot(random_weighting, np.kron(y[k1 + k2 + 2 + differs_at], u[k1 + k2 + 2]))
    #                    for u, y in trajectories], axis=0)
    #                for k1 in range(2*s + 1)] for k2 in tqdm(range(2*s + 1))])
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

def get_lds_parameters(component, m, n, p):
    component = component.reshape((-1, m, p))
    D = component[0]
    s = len(component)//2
    
    # Ho-Kalman
    H = np.block([[component[i+j+1] for j in range(s+1)] for i in range(s)])
    H_minus, H_plus = H[:, :p*s], H[:, -p*s:]
    U, S, Vh = la.svd(H_minus)
    U, S, Vh = U[:,:n], S[:n], Vh[:n,:]
    O, Q = U * np.sqrt(S), np.sqrt(S.reshape((-1, 1))) * Vh
    C, B = O[:m], Q[:,:p]
    A = la.pinv(O) @ H_plus @ la.pinv(Q)
    return A, B, C, D


# def get_R(trajectories):
#     return np.mean([np.einsum("j,tk->tkj", u[0], y) for u, y in trajectories], axis=0)

# def adjust_for_weights(G_tilde, R, s):
#     sqrt_weights, _, _, _ = la.lstsq(G_tilde.reshape((G_tilde.shape[0], -1)).T, R[:2*s+1].flatten())
#     return [Gi/sqrt_w for Gi, sqrt_w in zip(G_tilde, sqrt_weights)], sqrt_weights**2