import typing
import logging
import numpy as np
from time import time

from algorithms_package.src.contraction import cronMulVecL, partialContractionsRLKronecker
from algorithms_package.src import contraction, random_tensor_generation, pseudoinverse_primitives
from algorithms_package.src.random_tensor_generation import createRandomTensor
from algorithms_package.src import primitives

from joblib import Parallel, delayed


def preciseHadamardProduct(tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = []
    for i in range(len(tt_tensors1)):
        a, b = tt_tensors1[i], tt_tensors2[i]
        result_kernel = np.zeros((a.shape[0] * b.shape[0], a.shape[1], a.shape[2] * b.shape[2]), dtype=np.complex64)
        for i in range(a.shape[1]):
            result_kernel[:, i, :] = np.kron(a[:, i, :], b[:, i, :])
        answer.append(result_kernel)
    return answer


def generalizedApproximateHadamardProduct(
    tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array], random_tensor: np.array, func=lambda x: x
):
    answer = []
    contractions = partialContractionsRLKronecker(
        tt_tensors1L=tt_tensors1, tt_tensors1R=tt_tensors2, tt_tensors2=random_tensor
    )
    left_tensor = np.ones((1, 1, 1))
    size = len(tt_tensors1)

    for i in range(size):
        z = cronMulVecL(tt_tensors1[i], tt_tensors2[i], left_tensor)
        z = func(z)
        if i == size - 1:
            ans = np.transpose(z, (3, 2, 1, 0))
            ans = np.reshape(ans, (ans.shape[0], ans.shape[1], 1))
            answer.append(ans)
            return answer
        full = np.einsum('abcd,abe->dce', z, contractions[i + 1])

        y = full.reshape((-1, full.shape[-1]), order='F')
        q, _ = np.linalg.qr(y)
        ans = q.reshape(full.shape[:-1] + (q.shape[1],), order='F')
        answer.append(ans)
        left_tensor = np.einsum('abc,deba->dec', ans, z)
    return answer


def generalizedTwoSidedHadamardProduct(
    tt_tensors1: typing.List[np.array],
    tt_tensors2: typing.List[np.array],
    left_random_tensor: np.array,
    right_random_tensor: np.array,
    leave_left: bool,
):
    times_dict = {}
    time0 = time()
    left_contractions = contraction.partialContractionsLRKronecker(tt_tensors1, tt_tensors2, left_random_tensor)
    time1 = time()
    right_contractions = contraction.partialContractionsRLKronecker(tt_tensors1, tt_tensors2, right_random_tensor)
    time2 = time()
    times_dict['left_contraction'] = time1 - time0
    times_dict['right_contraction'] = time2 - time1
    psi_tensors, psi_times = contraction.countPsiTensorsKronecker(
        left_contractions, tt_tensors1, tt_tensors2, right_contractions
    )
    phi_tensors, phi_times = contraction.countPhiTensorsKronecker(left_contractions, right_contractions)
    answer, times = (
        pseudoinverse_primitives.processTensorsTakeLeft(psi_tensors, phi_tensors)
        if leave_left
        else pseudoinverse_primitives.processTensorsTakeRight(psi_tensors, phi_tensors)
    )
    times_dict['psi'] = psi_times
    times_dict['phi'] = phi_times
    times_dict['invert'] = times
    return answer, times_dict


def approximateHadamardProduct(
    tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array], desired_ranks: typing.List[int], seed: int
):
    modes = primitives.countModes(tt_tensors1)
    random_tensor = createRandomTensor(modes, desired_ranks, seed=seed)
    return generalizedApproximateHadamardProduct(tt_tensors1, tt_tensors2, random_tensor)


def approximateTwoSidedHadamardProduct(
    tt_tensors1: typing.List[np.array],
    tt_tensors2: typing.List[np.array],
    desired_ranks: typing.List[int],
    auxiliary_ranks: typing.List[np.array],
    seed: int,
    leave_left=True,
):
    modes = primitives.countModes(tt_tensors1)
    if not leave_left:
        desired_ranks, auxiliary_ranks = auxiliary_ranks, desired_ranks
    time0 = time()
    left_random_tensor = random_tensor_generation.createRandomTensor(modes, desired_ranks, seed)
    time1 = time()
    right_random_tensor = random_tensor_generation.createRandomTensor(modes, auxiliary_ranks, seed)
    time2 = time()
    answer, time_dict = generalizedTwoSidedHadamardProduct(
        tt_tensors1, tt_tensors2, left_random_tensor, right_random_tensor, leave_left
    )
    time_dict['left_random'] = time1 - time0
    time_dict['right_random'] = time2 - time1
    return answer, time_dict
