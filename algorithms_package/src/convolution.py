import typing
import numpy as np
from algorithms_package.src import contraction, primitives, hadamard_product, random_tensor_generation
from algorithms_package.src.random_tensor_generation import createRandomTensor
import logging


def countFourier(tt_tensor: typing.List[np.ndarray]):
    answer = []
    for i in range(len(tt_tensor)):
        answer.append(np.fft.fft(tt_tensor[i], axis=1))
    return answer


def countInverseFourier(tt_tensor: typing.List[np.ndarray]):
    answer = []
    for i in range(len(tt_tensor)):
        answer.append(np.fft.ifft(tt_tensor[i], axis=1))
    return answer


def paddingForConvolutionKernels(tt_kernel1: np.ndarray, tt_kernel2: np.ndarray):
    s = tt_kernel1.shape[1] + tt_kernel2.shape[1] - 1
    answer1 = np.zeros((tt_kernel1.shape[0], s, tt_kernel1.shape[2]))
    answer1[:, : tt_kernel1.shape[1], :] = tt_kernel1
    answer2 = np.zeros((tt_kernel2.shape[0], s, tt_kernel2.shape[2]))
    answer2[:, : tt_kernel2.shape[1], :] = tt_kernel2
    return answer1, answer2


def paddingForConvolutionTTTensors(tt_tensors1: typing.List[np.ndarray], tt_tensors2: typing.List[np.ndarray]):
    answer = list(map(lambda x: paddingForConvolutionKernels(x[0], x[1]), zip(tt_tensors1, tt_tensors2)))
    return [x[0] for x in answer], [x[1] for x in answer]


def primitiveCycleConvolutionMatrices(tt_tensor1: typing.List[np.ndarray], tt_tensor2: typing.List[np.ndarray]):
    assert len(tt_tensor1) == 2
    assert len(tt_tensor2) == 2
    matrix1 = np.einsum('ijk,klm->ijlm', tt_tensor1[0], tt_tensor1[1])[0, :, :, 0]
    matrix2 = np.einsum('ijk,klm->ijlm', tt_tensor2[0], tt_tensor2[1])[0, :, :, 0]
    answer = np.zeros(matrix1.shape)
    assert matrix1.shape == matrix2.shape
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            kek = np.roll(np.flip(matrix2), (i + 1, j + 1), axis=(0, 1))
            answer[i, j] = np.sum(matrix1 * kek)
    return answer


def preciseCycleConvolution(tt_tensor1: typing.List[np.ndarray], tt_tensor2: typing.List[np.ndarray]):
    return countInverseFourier(
        hadamard_product.preciseHadamardProduct(countFourier(tt_tensor1), countFourier(tt_tensor2))
    )


def preciseConvolution(tt_tensors1: typing.List[np.ndarray], tt_tensors2: typing.List[np.ndarray]):
    tt_tensors1_, tt_tensors2_ = paddingForConvolutionTTTensors(tt_tensors1, tt_tensors2)
    return preciseCycleConvolution(tt_tensors1_, tt_tensors2_)


def approximateCycleConvolution(
    tt_tensors1: typing.List[np.ndarray],
    tt_tensors2: typing.List[np.ndarray],
    desired_ranks: typing.List[int],
    seed: int,
):
    modes = primitives.countModes(tt_tensors1)
    random_tensor = createRandomTensor(modes, desired_ranks, seed)
    return hadamard_product.generalizedApproximateHadamardProduct(
        countFourier(tt_tensors1),
        countFourier(tt_tensors2),
        countInverseFourier(random_tensor),
        lambda z: np.fft.ifft(z, axis=2),
    )


def approximateConvolution(
    tt_tensors1: typing.List[np.ndarray],
    tt_tensors2: typing.List[np.ndarray],
    desired_ranks: typing.List[int],
    seed: int,
):
    tt_tensors1_, tt_tensors2_ = paddingForConvolutionTTTensors(tt_tensors1, tt_tensors2)
    return approximateCycleConvolution(tt_tensors1_, tt_tensors2_, desired_ranks, seed)


def approximateTwoSidedCycleConvolution(
    tt_tensors1: typing.List[np.ndarray],
    tt_tensors2: typing.List[np.ndarray],
    desired_ranks: typing.List[int],
    auxiliary_ranks: typing.List[int],
    seed: int,
    leave_left,
):
    modes = primitives.countModes(tt_tensors1)
    if not leave_left:
        desired_ranks, auxiliary_ranks = auxiliary_ranks, desired_ranks
    left_random_tensor = random_tensor_generation.createRandomTensor(modes, desired_ranks, seed)
    right_random_tensor = random_tensor_generation.createRandomTensor(modes, auxiliary_ranks, seed)
    p = hadamard_product.generalizedApproximateHadamardProduct(
        countFourier(tt_tensors1),
        countFourier(tt_tensors2),
        countInverseFourier(left_random_tensor),
        countInverseFourier(right_random_tensor),
        leave_left,
    )
    return countInverseFourier(p)


def approximateTwoSidedConvolution(
    tt_tensors1: typing.List[np.ndarray],
    tt_tensors2: typing.List[np.ndarray],
    desired_ranks: typing.List[int],
    auxiliary_ranks: typing.List[int],
    seed: int,
):
    tt_tensors1_, tt_tensors2_ = paddingForConvolutionTTTensors(tt_tensors1, tt_tensors2)
    return approximateTwoSidedCycleConvolution(tt_tensors1_, tt_tensors2_, desired_ranks, auxiliary_ranks, seed)
