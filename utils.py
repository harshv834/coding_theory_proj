import numpy as np


class Stats:
    def __init__(self, l1_dist, absolute_correct, is_codeword, n, rate, frac_of_errs):
        self.l1_dist = l1_dist
        self.absolute_correct = absolute_correct
        self.is_codeword = is_codeword
        self.n = n
        self.rate = rate
        self.frac_of_errs = frac_of_errs


def add_error(corr_y, num_errs):
    e = np.zeros(corr_y.shape)
    err_idx = np.random.choice(corr_y.shape[0], num_errs)
    e[err_idx] = 1
    return (corr_y + e) % 2
