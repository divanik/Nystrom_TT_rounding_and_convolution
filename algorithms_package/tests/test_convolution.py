import numpy as np
from algorithms_package.src import hadamard_product, contraction, primitives, convolution
from algorithms_package.src.random_tensor_generation import createRandomTensor, createExampleTensor
from algorithms_package.src import rounding
import logging


def test_precise_cycle_convolution_correctness():
    for i in range(100):
        modes = [5, 4]
        desired_ranks1 = np.random.randint(5, size=1) + 1
        example_tensor1 = createExampleTensor(
            modes, desired_ranks1, variance=1, seed=4 * i + 1)
        desired_ranks2 = np.random.randint(5, size=1) + 1
        example_tensor2 = createExampleTensor(
            modes, desired_ranks2, variance=1, seed=4 * i + 1)
        x1 = convolution.primitiveCycleConvolutionMatrices(
            example_tensor1, example_tensor2)
        x2 = primitives.countTensor(
            convolution.preciseCycleConvolution(example_tensor1, example_tensor2))
        assert primitives.tensorsRelativeComparance(x1, x2) < 1e-6


def test_approximate_cycle_convolution_correctness():
    for i in range(100):
        np.random.seed(4 * i)
        modes = np.random.randint(10, size=10) + 2
        desired_ranks1 = np.random.randint(5, size=9) + 1
        desired_ranks2 = np.random.randint(5, size=9) + 1
        desired_ranks = np.random.randint(6, size=9) + 2
        example_tensor1 = createExampleTensor(
            modes, desired_ranks1, variance=100, seed=4 * i + 1)
        example_tensor2 = createExampleTensor(
            modes, desired_ranks2, variance=100, seed=4 * i + 2)
        approximate_product = convolution.approximateCycleConvolution(
            example_tensor1, example_tensor2, desired_ranks, seed=4 * i + 3)
        precise_product = convolution.preciseCycleConvolution(
            example_tensor1, example_tensor2)
        precise_product_rounded = rounding.randomizeThenOrthogonalize(
            precise_product, desired_ranks=desired_ranks, seed=4 * i + 3)
        assert primitives.ttTensorsRelativeComparance(
            approximate_product, precise_product_rounded) < 1e-6


def test_approximate_convolution_correctness():
    for i in range(100):
        np.random.seed(4 * i)
        modes = np.random.randint(10, size=10) + 2
        desired_ranks1 = np.random.randint(5, size=9) + 1
        desired_ranks2 = np.random.randint(5, size=9) + 1
        desired_ranks = np.random.randint(6, size=9) + 2
        example_tensor1 = createExampleTensor(
            modes, desired_ranks1, variance=100, seed=4 * i + 1)
        example_tensor2 = createExampleTensor(
            modes, desired_ranks2, variance=100, seed=4 * i + 2)
        approximate_product = convolution.approximateConvolution(
            example_tensor1, example_tensor2, desired_ranks, seed=4 * i + 3)
        precise_product = convolution.preciseConvolution(
            example_tensor1, example_tensor2)
        precise_product_rounded = rounding.randomizeThenOrthogonalize(
            precise_product, desired_ranks=desired_ranks, seed=4 * i + 3)
        assert primitives.ttTensorsRelativeComparance(
            approximate_product, precise_product_rounded) < 1e-6
