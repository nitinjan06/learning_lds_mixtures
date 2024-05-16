from macros import *
from typing import Optional

@dataclass
class LDS:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    process_cov: Optional[np.ndarray] = None
    output_cov: Optional[np.ndarray] = None

    markov_params: list = field(default_factory=list)
    
    def get_observability(self, s):
        return np.concatenate([self.C @ mpow(self.A, i) for i in range(s)])
    
    def get_controllability(self, s):
        return np.concatenate([mpow(self.A, i) @ self.B for i in range(s)], axis = 1)

    def get_s(self):
        # can make this faster
        self.s = 1
        while rank(self.get_observability(self.s)) != self.n: self.s += 1
        while rank(self.get_controllability(self.s)) != self.n: self.s += 1

    def get_markov_param(self, i):
        # can make this faster
        while len(self.markov_params) <= i:
            self.markov_params.append(self.C @ mpow(self.A, len(self.markov_params)-1) @ self.B)
        
        return self.markov_params[i]

    def __post_init__(self):
        self.n, self.p = self.B.shape
        self.m = self.C.shape[0]
        assert self.A.shape == (self.n, self.n)
        assert self.C.shape == (self.m, self.n)
        assert self.D.shape == (self.m, self.p)

        if self.process_cov is None: self.process_cov = np.eye(self.n)
        if self.output_cov is None: self.output_cov = np.eye(self.m)
        self.markov_params.append(self.D)

        # check individual assumptions
        opA, opB, opC, opD = [op_norm(x) for x in [self.A, self.B, self.C, self.D]]
        # assert 1 <= opB and 1 <= opC
        self.kappa = max(opA, opB, opC, opD)

        self.get_s()

    def generate_trajectory(self, length):
        u, w, z = normal(size=(length, self.p)), multivariate_normal(np.zeros(self.n), self.process_cov, size=length), multivariate_normal(np.zeros(self.m), self.output_cov, size=length)
        x, y = [normal(size=self.n)], []
        for t in range(length):
            y.append(self.C @ x[t] + self.D @ u[t] + z[t])
            x.append(self.A @ x[t] + self.B @ u[t] + w[t])

        return u, np.array(y)
    
    def get_nll(self, trajectory):
        ans = np.double(0)
        
        u, y = trajectory
        x_mean, x_cov = np.zeros(self.n), np.eye(self.n)
        for t in range(len(u)):
            prefit_resid = y[t] - self.C @ x_mean - self.D @ u[t]
            prefit_cov = self.C @ x_cov @ self.C.T + self.output_cov
            prefit_cov_inv = la.inv(prefit_cov)
            nl_pdf = self.m/2 * np.log(2*np.pi) + np.log(la.det(prefit_cov))/2 + (prefit_resid @ prefit_cov_inv @ prefit_resid)/2
            ans += nl_pdf

            kalman_gain = x_cov @ self.C.T @ prefit_cov_inv
            postfit_x_mean = x_mean + kalman_gain @ prefit_resid
            postfit_x_cov = x_cov - kalman_gain @ self.C @ x_cov
            x_mean = self.A @ postfit_x_mean + self.B @ u[t]
            x_cov = self.A @ postfit_x_cov @ self.A.T + self.process_cov

        return ans