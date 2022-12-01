import numpy as np
import galois
GF = galois.GF(2)

class Stats:
    def __init__(self, l1_dist, absolute_correct, is_codeword, n, rate, frac_of_errs):
        self.l1_dist = l1_dist
        self.absolute_correct = absolute_correct
        self.is_codeword = is_codeword
        self.n = n
        self.rate = rate
        self.frac_of_errs = frac_of_errs


def add_error(corr_y, num_errs):
    e = GF.Zeros(corr_y.shape)
    err_idx = np.random.choice(corr_y.shape[0], num_errs)
    e[err_idx] = 1
    return (corr_y + e) 

def h_inv(h): 
    return (np.log2(1 + np.sqrt(1-np.power(h, 4/3))) + (h-1))/(2*np.arctanh(np.sqrt(1-np.power(h,4/3))))

def h_inv_alt(h): 
    return h/(2*np.log2(6/h))

