import numpy as np


class Stats:
    def __init__(self, l1_dist, absolute_correct, is_codeword):
        self.l1_dist = l1_dist
        self.absolute_correct = absolute_correct
        self.is_codeword = is_codeword


def add_error(corr_y, num_errs):
    e = np.zeros(corr_y.shape)
    err_idx = np.random.choice(corr_y.shape[0], num_errs)
    e[err_idx] = 1
    return (corr_y + e) % 2
