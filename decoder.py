import numpy as np
from tqdm import tqdm
from abc import ABC


METRIC_CHOICES = ["sat", "unsat", "unsat_sat"]
SELECTOR_CHOICES = ["greedy", "weighted"]


class BaseAlgo(ABC):
    def __init__(self, algo_name, algo_params={}):
        super(BaseAlgo, self).__init__()
        self.algo_name = algo_name
        self.algo_params = algo_params

    def decode(self, H, y_err):
        raise NotImplementedError


class BitFlipAlgo(BaseAlgo):
    def __init__(self, algo_name, algo_params={}):
        super(BitFlipAlgo, self).__init__(algo_name=algo_name, algo_params=algo_params)

    def bit_flip(self, H, y_err):
        metric = self.algo_params["metric"](H, y_err)
        bit_flip_idx = self.algo_params["selector"](metric, y_err)
        new_y = y_err
        new_y[bit_flip_idx] = (new_y[bit_flip_idx] + 1) % 2
        return new_y

    def decode(self, H, y_err):
        y_list = [y_err]
        for i in range(self.algo_params["max_iter"]):
            syndrome = (H @ y_list[i]) % 2
            if (syndrome == 0).all():
                return y_list
            else:
                y_list.append(self.bit_flip(H, y_list[i]))
        return y_list


class Metric:
    def __init__(self, name):
        self.name = name
        assert self.name in [
            "sat",
            "unsat",
            "unsat_sat",
        ], "metric {} not implemented".format(self.name)

    def __call__(self, H, y_err):
        syndrome = (H @ y_err).astype(int) % 2
        if self.name in ["sat", "unsat_sat"]:
            satisfied_parity_idx = np.arange(H.shape[0])[syndrome != 0]
            num_satisfied = H[satisfied_parity_idx].sum(axis=0)
        if self.name in ["unsat", "unsat_sat"]:
            unsatisfied_parity_idx = np.arange(H.shape[0])[syndrome != 0]
            num_unsatisfied = H[unsatisfied_parity_idx].sum(axis=0)
        if self.name == "sat":
            return -num_satisfied
        elif self.name == "unsat":
            return num_unsatisfied
        else:
            return num_unsatisfied - num_satisfied


class BitSelector:
    def __init__(self, name, params={"lambda": 1}):
        self.name = name
        self.params = params
        assert self.name in [
            "greedy",
            "weighted",
        ], "Bit selection method {} not implemented".format(self.name)

    def __call__(self, metric, y_err):
        if self.name == "greedy":
            return np.argmax(metric)
        else:
            probs = np.exp(self.params["lambda"] * metric)
            probs = probs / probs.sum()
            return np.random.choice(np.arange(metric.shape[0]), p=probs)


class ExploreSelector(BitSelector):
    def __init__(self, name, params={"lambda": 1, "p": 0.2}):
        super(ExploreSelector, self).__init__(name=name, params=params)

    def __call__(self, metric, y_err):
        if np.random.rand() <= self.params["p"]:
            return np.random.choice(np.arange(metric.shape[0]))
        else:
            return super().__call__(metric, y_err)


# class ExploreComp(BitFlipAlgo):
#     def __init__(self, algo_name, algo_params = {}):
#         super(ExploreComp, self).__init__(algo_name=algo_name, algo_params=algo_params)

#     def bit_flip(self, y_err):
#         if np.random.rand() < self.algo_params["p"]:
#             bit_flip_idx = np.random.choice(y_err.shape[0])
#             new_y = y_err
#             new_y[bit_flip_idx] = (new_y[bit_flip_idx]+1) % 2
#             return new_y
#         else:
#             return super().bit_flip(y_err)

# metric =


# def message_v_c(i, prior, check_ll, H):
#     neighbor_i = H[:,i] * check_ll
#     neighbor_i[i] = prior[i]
#     return neighbor_i.sum()

# def message_c_v(i, check_ll, H)

# def belief_propagation(H, y, prior, max_iter=int(1e3)):
#     m,n = H.shape
#     node_ll = np.zeros(H.shape[1])
#     check_ll = np.zeros(H.shape[0])
#     prior_ll = np.log(prior / (1 - prior))
#     node_hist = np.zeros(node_ll.shape)
#     check_hist = np.zeros(check_ll.shape)
#     for iter in tqdm(range(max_iter)):
#         node_hist = node_ll
#         check_hist = check_ll
#         for i in range(n):
#             neighbor_i = H[:,]
#             node_ll  =
