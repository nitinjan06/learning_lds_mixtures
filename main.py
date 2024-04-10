from macros import *
from true_params import *
from learn_mixture import *

def array_printer(arr):
    for line in arr:
        print(str(line).replace("\n", ""))

if __name__ == "__main__":
    s = max(lds.s for lds in LDS_LIST)
    kappa = max(*[lds.kappa for lds in LDS_LIST],
                *[op_norm(lds.get_observability(2*s))/min_sv(lds.get_controllability(s)) for lds in LDS_LIST])
    
    print(f"s={s}, kappa={kappa}")

    expected_weights = [0.5, 0.5]
    samples = generate_samples(LDS_LIST, expected_weights, s, 10000)

    Pi_M = get_Pi_M(samples, s)
    G_tilde = get_G_tilde(p, n, s, k, Pi_M)
    R = get_R(samples)
    approx_weights, markov_parameters = adjust_for_weights(G_tilde, R, s)

    expected_Pi_M = get_expected_Pi_M(LDS_LIST, expected_weights, s)

    np.set_printoptions(formatter={"float": lambda x: "{0:+0.1f}".format(x)})
    array_printer(Pi_M)
    print()
    array_printer(expected_Pi_M)
    print()
    array_printer(Pi_M - expected_Pi_M)