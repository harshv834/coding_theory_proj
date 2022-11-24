import numpy as np
from tqdm import tqdm


def bit_flip_dec(H, y, max_iter=int(1e3)):
    curr_y = y
    for _ in tqdm(range(max_iter)):
        syndrome = H @ curr_y
        unsatisfied_parity_idx = np.arange(H.shape[0])[syndrome != 0]
        if syndrome == np.zeros(y.shape[0]):
            return curr_y
        else:
            satisfied_parity_idx = np.arange(H.shape[0])[syndrome == 0]
            num_unsatisfied = H[unsatisfied_parity_idx].sum(axis=0)
            num_satisfied = H[satisfied_parity_idx].sum(axis=0)
            bit_flip_idx = np.argmax(num_unsatisfied - num_satisfied)
            curr_y[bit_flip_idx] = (curr_y[bit_flip_idx] + 1) % 2
    return curr_y


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
