import numpy as np
from tqdm import tqdm
from abc import ABC
from utils import GF

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

    def bit_flip(self, H, y_err, syndrome):
        metric = self.algo_params["metric"](H, y_err, syndrome)
        bit_flip_idx = self.algo_params["selector"](metric, y_err)
        new_y = y_err
        new_y[bit_flip_idx] += GF(1)
        return new_y, bit_flip_idx

    def decode(self, H, y_err):
        y_list = [y_err]
        syndrome = H @ y_list[0]
        for i in range(self.algo_params["max_iter"]):
            if (syndrome == 0).all():
                return y_list[-1]
            else:
                y_new, bit_flip_idx = self.bit_flip(H, y_list[i], syndrome)
                y_list.append(y_new)
                syndrome += H[:, bit_flip_idx]
        return y_list[-1]

class BitFlipTopL(BitFlipAlgo):
    def __init__(self, algo_name, algo_params = {}):
        super(BitFlipTopL, self).__init__(algo_name, algo_params)
    
    def decode(self, H, y_err, half_min_dist):
        topL_list = [y_err]
        flag=False
        for _ in range(self.algo_params["max_iter"]):
            syndromes = [H @ y for y in topL_list]
            all_candidates = []
            for j in range(len(topL_list)):
                if (syndromes[j] == 0).all():
                    return topL_list[j]
            
                metric = self.algo_params["metric"](H, topL_list[j], syndromes[j])
                sort_idx = np.argsort(metric)
                bits_to_flip = sort_idx[:self.algo_params["L"]]
                candidates = []
                for bit_idx in bits_to_flip:
                    new_y = topL_list[j].copy()
                    new_y[bit_idx] += GF(1)
                    candidates.append(new_y)
                all_candidates += candidates
            candidates_to_keep = [code for code in all_candidates if np.array(np.abs(code - y_err)).sum() <= half_min_dist]
            if len(candidates_to_keep) == 0:
                flag = True
            else:
                all_candidates = candidates_to_keep
            # all_candidates = [code for code in all_candidates if np.array(np.abs(code - y_err)).sum() < half_min_dist]
            scores = np.array([self.code_metric(H, y, y_err) for y in all_candidates])
            if flag:
                best_idx = np.argmin(scores)
                best_code = topL_list[best_idx]
                return best_code
            if len(all_candidates) > self.algo_params["L"]:
                sort_idx = np.argsort(scores)
                scores = scores[sort_idx[:self.algo_params["L"]]]
                topL_list_temp = [all_candidates[j] for j in sort_idx.astype(int)[:self.algo_params["L"]]]
                topL_temp_set = set([tuple(np.array(code)) for code in topL_list_temp])
                topL_set = set([tuple(np.array(code)) for code in topL_list])
                if len(topL_temp_set.intersection(topL_set)) >= self.algo_params["coeff"]*self.algo_params["L"]:
                    break
                topL_list = topL_list_temp
            else:
                topL_list = all_candidates

        best_idx = np.argmin(scores)
        best_code = topL_list[best_idx]
        return best_code

    def code_metric(self, H, y_curr, y_orig):
        syndrome = H @ y_curr
        num_unsatisfied_constr = (np.array(syndrome) != 0).sum()
        dist_from_orig = np.abs(np.array(y_curr - y_orig)).sum()
        wt = 0.2
        return num_unsatisfied_constr + wt*dist_from_orig

        
    
        





class Metric:
    def __init__(self, name):
        self.name = name
        assert self.name in [
            "sat",
            "unsat",
            "unsat_sat",
        ], "metric {} not implemented".format(self.name)

    def __call__(self, H, y_err, syndrome):
        if self.name in ["sat", "unsat_sat"]:
            satisfied_parity_idx = np.arange(H.shape[0])[syndrome == 0]
            num_satisfied = np.array(H)[satisfied_parity_idx].sum(axis=0)
        if self.name in ["unsat", "unsat_sat"]:
            unsatisfied_parity_idx = np.arange(H.shape[0])[syndrome != 0]
            num_unsatisfied = np.array(H)[unsatisfied_parity_idx].sum(axis=0)

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
