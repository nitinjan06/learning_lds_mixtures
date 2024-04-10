from macros import *

def get_Pi_M_with_var(trajectories, s):
    Ts = [np.block([[np.outer(np.kron(u[0], y[k1]), np.kron(u[k1+1], y[k1 + k2 + 1])) for k1 in range(2*s+1)] for k2 in range(2*s+1)]) for u, y in trajectories]
    T_mean = np.mean(Ts, axis=0)
    T_var = np.var(Ts, axis=0)
    # doesn't symmetrize
    return T_mean, T_var
 
def get_Pi_M(trajectories, s):
    T_k1k2 = [[
                np.mean([
                    np.outer(np.kron(y[k1 + k2 + 1], u[k1+1]), np.kron(y[k1], u[0]))
                for u, y in trajectories], axis=0)
             for k1 in range(2*s + 1)] for k2 in range(2*s + 1)]
    Pi_M = np.block(T_k1k2)
    symmetrized_Pi_M = (Pi_M + Pi_M.T)/2 # only works for real matrices
    return symmetrized_Pi_M

def get_G_tilde(p, n, s, k, Pi_M): # assumes Pi_M is symmetrized
    U, S, Vh = la.svd(Pi_M, hermitian=True)
    truncated_S, truncated_U, truncated_Vh = S[:k], U[:,:k], Vh[:k,:]
    print("check:", np.array_equal(truncated_U, truncated_Vh.T))
    
    components = truncated_U * np.sqrt(truncated_S)

    # test
    approx_Pi_M = components @ components.T
    print("frobenius norm of Pi_M:", la.norm(Pi_M))
    print("frobenius norm after approx:", la.norm(Pi_M - approx_Pi_M))

    G = components.T.reshape((k, 2*s+1, p, n))
    return G

def get_R(trajectories):
    return np.mean([np.einsum("j,tk->tkj", u[0], y) for u, y in trajectories], axis=0)

def adjust_for_weights(G_tilde, R, s):
    sqrt_weights, _, _, _ = la.lstsq(G_tilde.reshape((G_tilde.shape[0], -1)).T, R[:2*s+1].flatten())
    return [Gi/sqrt_w for Gi, sqrt_w in zip(G_tilde, sqrt_weights)], sqrt_weights**2