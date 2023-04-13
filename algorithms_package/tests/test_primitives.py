import numpy as np
from algorithms_package.src import primitives
from algorithms_package.src import random_tensor_generation


def testFastFrobeniusDistance():
    modes = [3, 6, 5, 3]
    rank1 = [4, 7, 2]
    rank2 = [3, 6, 9]
    tt = random_tensor_generation.createExampleTensor(modes, rank1, variance=100)
    tt2 = random_tensor_generation.createExampleTensor(modes, rank2, variance=100)
    assert primitives.countFrobeniusDistance(tt, tt) < 1e-3
    assert np.abs(primitives.countFrobeniusDistance(tt, tt2) -
                  primitives.countFrobeniusDistanceSlow(tt, tt2)) < 1e-3
