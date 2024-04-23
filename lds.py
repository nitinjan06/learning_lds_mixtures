from macros import *

@dataclass
class LDS:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
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

        self.markov_params.append(self.D)

        # check individual assumptions
        opA, opB, opC, opD = [op_norm(x) for x in [self.A, self.B, self.C, self.D]]
        assert 1 <= opB and 1 <= opC
        self.kappa = max(opA, opB, opC, opD)

        self.get_s()

    def generate_trajectory(self, length):
        u, w, z = normal(size=(length, self.p)), normal(size=(length, self.n)), normal(size=(length, self.m))
        x, y = [normal(size=self.n)], []
        for t in range(length):
            y.append(self.C @ x[t] + self.D @ u[t] + z[t])
            x.append(self.A @ x[t] + self.B @ u[t] + w[t])

        return u, np.array(y)