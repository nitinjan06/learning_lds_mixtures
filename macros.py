from functools import partial
from dataclasses import dataclass, field
import numpy as np
import numpy.linalg as la
normal = np.random.normal
op_norm = partial(la.norm, ord=2)
min_sv = partial(la.norm, ord=-2)
rank = la.matrix_rank
mpow = la.matrix_power

def get_r2(pred, goal):
    return 1-la.norm(goal - pred)**2/la.norm(goal)**2