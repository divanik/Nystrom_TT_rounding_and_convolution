import numpy as np
from algorithms_package.src import contraction, hadamard_product, random_tensor_generation


def test_partial_contraction_running_correctness():
    for _ in range(100):
        modes = np.random.randint(10, size=10) + 2
        desired_ranks1 = np.random.randint(5, size=9) + 1
        random_tensor1 = random_tensor_generation.createRandomTensor(modes, desired_ranks1, seed=0)
        desired_ranks2 = np.random.randint(5, size=9) + 1
        random_tensor2 = random_tensor_generation.createRandomTensor(modes, desired_ranks2, seed=1)
        matricesRL = contraction.partialContractionsRL(random_tensor1, random_tensor2)
        matricesLR = contraction.partialContractionsLR(random_tensor1, random_tensor2)
        assert len(matricesLR) == len(matricesRL)
        scalar_products = []
        for i in range(len(matricesLR)):
            scalar_products.append(np.sum(matricesLR[i] * matricesRL[i]))
        assert max(scalar_products) - min(scalar_products) < 1e-12
