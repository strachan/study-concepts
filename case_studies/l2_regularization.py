import numpy as np
import sys

sys.path.append('..')
from utils import time_func_rep

W = np.random.randn(1000, 1)


@time_func_rep
def l2_reg_for(W):
    penalty = 0
    for i in np.arange(0, W.shape[0]):
        for j in np.arange(0, W.shape[1]):
            penalty += (W[i][j]**2)
    return penalty


@time_func_rep
def l2_reg_opt(W):
    penalty = np.sum(W ** 2)
    return penalty


result_for = l2_reg_for(W)
result_opt = l2_reg_opt(W)
assert abs(result_for - result_opt) < 0.00001
print(f'Penalty value is {result_opt}')
