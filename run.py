from decoder import (
    BitSelector,
    Metric,
    BitFlipAlgo,
    ExploreSelector,
    METRIC_CHOICES,
    SELECTOR_CHOICES,
)
from utils import Stats, add_error
from tqdm import tqdm
import itertools
import numpy as np
from pyldpc import make_ldpc, encode
import pickle
from multiprocessing.pool import ThreadPool as Pool

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


def test_code(code, n, rate, frac_of_errs, algorithm, trials_per_code, snr):
    d_c = int(n / 5)
    d_v = int((1 - rate) * d_c)
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    k = G.shape[1]
    v = np.random.randint(2, size=k)
    y = encode(G, v, snr)

    l1_dist = 0
    num_abs_correct = 0
    num_valid_codeword = 0

    for trial in range(trials_per_code):
        y_err = add_error(y, int(frac_of_errs * n))

        y_list = algorithm.decode(H, y_err)
        decode_y = y_list[-1]
        y = y.astype(int)
        decode_y = decode_y.astype(int)
        l1_dist += np.absolute(decode_y - y).sum()
        num_abs_correct += 1 if (decode_y == y).all() else 0
        num_valid_codeword += 1 if (H @ decode_y).all() == 0 else 0
    return np.array([l1_dist, num_abs_correct, num_valid_codeword])


def benchmark_algo(algorithm, codes_per_case=1, trials_per_code=1):
    stats = {}
    n = [1000]
    rate = [0.1, 0.25, 0.4]
    frac_of_errs = [0.1, 0.25, 0.4]
    cases = list(itertools.product(n, rate, frac_of_errs))
    ### This d_c and d_v can be used to set the code rate. I think d_c/n should be the rate.
    snr = 200
    p = Pool(10)
    for case in tqdm(cases):
        n, rate, frac_of_errs = case
        test_code_ = lambda code: test_code(
            code, n, rate, frac_of_errs, algorithm, trials_per_code, snr
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
    stats_dict = {}
    for metric_name, selector_name in itertools.product(
        METRIC_CHOICES, SELECTOR_CHOICES
    ):
        for i in range(2):
            metric = Metric(metric_name)
            num_trials_per_code = 5
            num_codes = 100
            algo_name = metric_name + "_" + selector_name
            if i == 1:
                algo_name += "_" + "explore"
                selector = ExploreSelector(selector_name)
            else:
                selector = BitSelector(selector_name)
            algo_params = {"max_iter": int(1e3), "metric": metric, "selector": selector}
            algo = BitFlipAlgo(algo_name=algo_name, algo_params=algo_params)
            stats = benchmark_algo(algo, num_codes, num_trials_per_code)
            stats_dict[algo_name] = stats
    with open("results.pickle", "wb") as f:
        pickle.dump(stats_dict, f)


if __name__ == "__main__":
    main()
