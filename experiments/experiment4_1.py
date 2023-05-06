import numpy as np
import tt
import experiments_stuff
from bilinear_package.src import primitives
from bilinear_package.src.hadamard_product import generalizedApproximateHadamardProduct
from bilinear_package.src.convolution import (
    approximateConvolution,
    preciseConvolution,
    countFourier,
    countInverseFourier,
    approximateCycleConvolution,
)
import time


from newton_experiment_primitives import create_exponential_grid, create_newtonial_potential_grid

from bilinear_package.src.convolution import preciseCycleConvolution
from newton_experiment_primitives import interpolateTTTensor, compressTTTensor
import pandas as pd

D = 40

df_time = pd.DataFrame()
df_precision = pd.DataFrame()

for d in range(3, 6):
    for grid_side in np.logspace(7, 10, 7, base=2, dtype=int):
        grid_size = 8 * (grid_side // 8)
        h = (2 * D) / grid_side
        tensor1 = create_exponential_grid(d, grid_side + 2, D + h / 2)
            tensor2 = create_newtonial_potential_grid(d, 2 * grid_side + 2, h * grid_side + h / 2)
            tensor1 = primitives.twoSidedPaddingTTTensor(tensor1, [(0, grid_side) for _ in range(d)])
        print(tt.vector.from_list(tensor1))
        print(tt.vector.from_list(tensor2))
        precise = preciseCycleConvolution(tensor1, tensor2)
        precise = primitives.twoSidedCuttingTTTensor(precise, [(0, grid_side + 1) for _ in range(d)])
        precise = tt.vector.from_list(precise)
        precise_norm = tt.vector.norm(precise)
        time1 = time.time()
        answer1 = approximateCycleConvolution(
            tensor1, tensor2, desired_ranks=np.ones(d - 1, dtype=int) * 30, seed=271 * d + grid_side
        )
        answer1 = primitives.twoSidedCuttingTTTensor(answer1, [(0, grid_side + 1) for _ in range(d)])
        time2 = time.time()
        answer2 = precise.round(0, 30)
        time3 = time.time()
        tensor1_ = tt.vector.from_list(countFourier(tensor1))
        tensor2_ = tt.vector.from_list(countFourier(tensor2))
        multifunc = tt.multifuncrs2([tensor1_, tensor2_], lambda x: x[:, 0] * x[:, 1], eps=1e-6, verb=0)
        answer3 = countInverseFourier(tt.vector.to_list(multifunc))
        answer3 = primitives.twoSidedCuttingTTTensor(answer1, [(0, grid_side + 1) for _ in range(d)])
        time4 = time.time()
        answer4 = approximateCycleConvolution(
            tensor1, tensor2, desired_ranks=np.ones(d - 1, dtype=int) * 30, seed=271 * d + grid_side
        )
        answer4 = primitives.twoSidedCuttingTTTensor(answer4, [(0, grid_side + 1) for _ in range(d)])
        time5 = time.time()
        df_time[f"{d}_{grid_side}"] = [time2 - time1, time3 - time2, time4 - time3]
        precision1 = tt.vector.norm(precise - tt.vector.from_list(answer1)) / precise_norm
        precision2 = tt.vector.norm(precise - answer2) / precise_norm
        precision3 = tt.vector.norm(precise - tt.vector.from_list(answer3)) / precise_norm
        df_precision[f"{d}_{grid_side}"] = [precision1, precision2, precision3]
        df_time.to_csv("experiments_results/experiment4/time")
        df_precision.to_csv("experiments_results/experiment4/precision")
        print(f"{d}_{grid_side}")
