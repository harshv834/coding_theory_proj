from decoder import (
    BitSelector,
    Metric,
    BitFlipTopL,
    METRIC_CHOICES
)
from utils import Stats, add_error, GF, h_inv_lb, h_inv_ub
from tqdm import tqdm
import itertools
import numpy as np
from pyldpc import make_ldpc, encode
import pickle
from multiprocessing.pool import ThreadPool as Pool
import torch

# _func = None


# def worker_init(func):
#     global _func
#     _func = func


# def worker(x):
#     return _func(x)


# def xmap(func, iterable, processes=None, args=None):
#     with Pool(
#         processes,
#         initializer=worker_init,
#         initargs=(func,),
#     ) as p:
#         return p.map(worker, iterable)


def test_code(code, n, frac_of_errs, algorithm, trials_per_code, snr, G, H, k, half_min_dist):
    v = np.random.randint(2, size=k)
    y = encode(G, v, snr)
    y = (y + 1) / 2
    y = y.astype(int).view(GF)
    H = H.astype(int).view(GF)
    #print(y, " y ")

    l1_dist = 0
    num_abs_correct = 0
    num_valid_codeword = 0
    # print( y,"y")

    for trial in range(trials_per_code):
        y_err = add_error(y, max(1, int(frac_of_errs * n)))
        # print(y_err,"y_err")
        decode_y = algorithm.decode(H, y_err, half_min_dist)
        # print(decode_y, "decoded y")
        l1_dist += np.absolute(np.array(decode_y - y)).sum()
        num_abs_correct += 1 if (decode_y == y).all() else 0
        num_valid_codeword += 1 if (H @ decode_y).all() == 0 else 0
    # print("l1 dist", l1_dist)
    # print("num_valid_codeword", num_valid_codeword)
    # print("num_abs_correct", num_abs_correct)
    
    return np.array([l1_dist, num_abs_correct, num_valid_codeword])


def benchmark_algo(p, algorithm, codes_per_case=1, trials_per_code=1):
    stats = {}
    n = [20, 50, 100]
    # n = [50]
    rate = [0.1, 0.25, 0.4]
    cases = list(itertools.product(n, rate))
    ### This d_c and d_v can be used to set the code rate. I think d_c/n should be the rate.
    snr = 200
    for case in tqdm(cases):
        n, rate = case
        d_c = 5
        d_v = int((1 - rate) * d_c)
        H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
        k = G.shape[1]
        # min_dist = get_min_dist(G)
        # algo_params = 
        frac_of_errs = h_inv_lb(1-rate)
        half_min_dist = max(1, int(h_inv_ub(1-rate)*n))
        test_code_ = lambda code: test_code(
            code, n, frac_of_errs, algorithm, trials_per_code, snr, G, H, k, half_min_dist
        )
        # with Pool(processes=10) as p:
        results = list(p.map(test_code_, np.arange(codes_per_case)))
        # results = list(xmap(test_code_, np.arange(codes_per_case), processes=10))
        results = np.vstack(results).sum(axis=0)

        stats[case] = Stats(
            results[0] / (trials_per_code * codes_per_case),
            results[1] / (trials_per_code * codes_per_case),
            results[2] / (trials_per_code * codes_per_case),
            n,
            rate,
            frac_of_errs,
        )

    return stats


def main():
    p = Pool(10)
    stats_dict = {}

    for metric_name in METRIC_CHOICES:
    # metric_name = "unsat_sat"
        selector_name = "greedy"
        metric = Metric(metric_name)
        num_trials_per_code = 2
        num_codes = 10
        L = 10
        algo_name = metric_name + "_" + selector_name + "_" + str(L)
        selector = BitSelector(selector_name)
        algo_params = {"max_iter": int(1e3), "metric": metric, "selector": selector, "L" : L, "coeff":1}
        algo_topL = BitFlipTopL(algo_name=algo_name, algo_params=algo_params)
        stats_dict[algo_name] = benchmark_algo(p, algo_topL, num_codes, num_trials_per_code)

    with open(f"results_top_10.pickle", "wb") as f:
        pickle.dump(stats_dict, f)


if __name__ == "__main__":
    main()
