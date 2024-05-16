from macros import *

def generate_mixture_samples(lds_list, weights, s, num_trajectories = 100):
    length = 6*s + 3
    sample_from = np.random.choice(lds_list, size=num_trajectories, p=weights)
    trajectories = [lds.generate_trajectory(length) for lds in sample_from]
    return trajectories