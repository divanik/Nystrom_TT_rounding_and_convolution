import typing
import logging
import numpy as np
import time

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
    n_jobs: int,
):
    if n_jobs == 1:
        left_contractions = contraction.partialContractionsLRKronecker(tt_tensors1, tt_tensors2, left_random_tensor)
        right_contractions = contraction.partialContractionsRLKronecker(tt_tensors1, tt_tensors2, right_random_tensor)
    else:
        contractions = Parallel(n_jobs=2)(
            [
                delayed(contraction.partialContractionsLRKronecker)(tt_tensors1, tt_tensors2, left_random_tensor),
                delayed(contraction.partialContractionsRLKronecker)(tt_tensors1, tt_tensors2, right_random_tensor),
            ]
        )
        left_contractions = contractions[0]
        right_contractions = contractions[1]
    psi_tensors = contraction.countPsiTensorsKronecker(
        left_contractions, tt_tensors1, tt_tensors2, right_contractions, n_jobs
    )
    phi_tensors = contraction.countPhiTensorsKronecker(left_contractions, right_contractions, n_jobs)
    return (
        pseudoinverse_primitives.processTensorsTakeLeft(psi_tensors, phi_tensors, n_jobs)
        if leave_left
        else pseudoinverse_primitives.processTensorsTakeRight(psi_tensors, phi_tensors, n_jobs)
    )


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
    n_jobs=1,
):
    modes = primitives.countModes(tt_tensors1)
    if not leave_left:
        desired_ranks, auxiliary_ranks = auxiliary_ranks, desired_ranks
    left_random_tensor = random_tensor_generation.createRandomTensor(modes, desired_ranks, seed)
    right_random_tensor = random_tensor_generation.createRandomTensor(modes, auxiliary_ranks, seed)
    return generalizedTwoSidedHadamardProduct(
        tt_tensors1, tt_tensors2, left_random_tensor, right_random_tensor, leave_left, n_jobs
    )
