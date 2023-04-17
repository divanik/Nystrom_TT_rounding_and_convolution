import typing
import logging
import numpy as np
from algorithms_package.src import primitives
from joblib import Parallel, delayed

from numpy.linalg import lstsq


def partialContractionsRL(tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = [np.ones((1, 1))]
    for tt1, tt2 in zip(reversed(tt_tensors1), reversed(tt_tensors2)):
        last = answer[-1]
        answer.append(np.einsum('ijk,kl,mjl->im', tt1, last, tt2))
    return list(reversed(answer))


def partialContractionsLR(tt_tensors1: typing.List[np.array], tt_tensors2: typing.List[np.array]):
    answer = [np.ones((1, 1))]
    for tt1, tt2 in zip(tt_tensors1, tt_tensors2):
        last = answer[-1]
        answer.append(np.einsum('ijk,il,ljm->km', tt1, last, tt2))
    return answer


def cronMulVecR(a: np.array, b: np.array, c: np.array):
    p = np.einsum('dbe,cel->dbcl', b, c)
    return np.einsum('abc,dbcl->adbl', a, p)


def cronMulVecL(a: np.array, b: np.array, c: np.array):
    p = np.einsum('ebd,cel->dbcl', b, c)
    return np.einsum('cba,dbcl->adbl', a, p)


def cronMulVecReduceModeR(a: np.array, b: np.array, c: np.array):
    p = np.einsum('dbe,cebl->dcbl', b, c)
    return np.einsum('abc,dcbl->adl', a, p)


def cronMulVecReduceModeL(a: np.array, b: np.array, c: np.array):
    p = np.einsum('ebd,cebl->cdbl', b, c)
    return np.einsum('cba,cdbl->adl', a, p)


def partialContractionsRLKronecker(
    tt_tensors1L: typing.List[np.array], tt_tensors1R: typing.List[np.array], tt_tensors2: typing.List[np.array]
):
    answer = [np.ones((1, 1, 1))]
    for tt1L, tt1R, tt2 in zip(reversed(tt_tensors1L), reversed(tt_tensors1R), reversed(tt_tensors2)):
        p = cronMulVecR(tt1L, tt1R, answer[-1])
        answer.append(np.einsum('ldu,abdu->abl', tt2, p))
    return list(reversed(answer))


def partialContractionsLRKronecker(
    tt_tensors1L: typing.List[np.array], tt_tensors1R: typing.List[np.array], tt_tensors2: typing.List[np.array]
):
    answer = [np.ones((1, 1, 1))]
    for tt1L, tt1R, tt2 in zip(tt_tensors1L, tt_tensors1R, tt_tensors2):
        p = cronMulVecL(tt1L, tt1R, answer[-1])
        answer.append(np.einsum('udl,abdu->abl', tt2, p))
    return answer


def _countPsiTensor(left_tensor: np.array, tt_kernel: np.array, right_tensor: np.array):
    p = np.einsum('ae,abc->ebc', left_tensor, tt_kernel)
    return np.einsum('ebc,cd->ebd', p, right_tensor)


def _countPhiTensor(left_tensor: np.array, right_tensor: np.array):
    return np.einsum('ax,ay->xy', left_tensor, right_tensor)


def _countPsiTensorKronecker(left_tensor: np.array, tt_kernel1: np.array, tt_kernel2: np.array, right_tensor: np.array):
    p = np.einsum('abe,anc->ebnc', left_tensor, tt_kernel1)
    q = np.einsum('bnd,cdf->cbnf', tt_kernel2, right_tensor)
    return np.einsum('ebnc,cbnf->enf', p, q)


def _countPhiTensorKronecker(left_tensor: np.array, right_tensor: np.array):
    return np.einsum('abx,aby->xy', left_tensor, right_tensor)


def countPsiTensors(
    left_contractions: typing.List[np.array],
    tt_tensor: typing.List[np.array],
    right_contractions: typing.List[np.array],
    n_jobs,
):
    modes = primitives.countModes(tt_tensor)
    return (
        Parallel(n_jobs=n_jobs)(
            delayed(_countPsiTensor)(left_contractions[i], tt_tensor[i], right_contractions[i + 1])
            for i in range(len(modes))
        )
        if n_jobs > 1
        else [_countPsiTensor(left_contractions[i], tt_tensor[i], right_contractions[i + 1]) for i in range(len(modes))]
    )


def countPsiTensorsKronecker(
    left_contractions: typing.List[np.array],
    tt_tensor1: typing.List[np.array],
    tt_tensor2: typing.List[np.array],
    right_contractions: typing.List[np.array],
    n_jobs,
):
    modes = primitives.countModes(tt_tensor1)
    return (
        Parallel(n_jobs=n_jobs)(
            delayed(_countPsiTensorKronecker)(
                left_contractions[i], tt_tensor1[i], tt_tensor2[i], right_contractions[i + 1]
            )
            for i in range(len(modes))
        )
        if n_jobs > 1
        else [
            _countPsiTensorKronecker(left_contractions[i], tt_tensor1[i], tt_tensor2[i], right_contractions[i + 1])
            for i in range(len(modes))
        ]
    )


def countPhiTensors(left_contractions: typing.List[np.array], right_contractions: typing.List[np.array], n_jobs):
    return (
        Parallel(n_jobs=n_jobs)(
            delayed(_countPhiTensor)(left_contractions[i], right_contractions[i])
            for i in range(1, len(left_contractions) - 1)
        )
        if n_jobs > 1
        else [
            _countPhiTensor(left_contractions[i], right_contractions[i]) for i in range(1, len(left_contractions) - 1)
        ]
    )


def countPhiTensorsKronecker(
    left_contractions: typing.List[np.array], right_contractions: typing.List[np.array], n_jobs
):
    return (
        Parallel(n_jobs=n_jobs)(
            delayed(_countPhiTensorKronecker)(left_contractions[i], right_contractions[i])
            for i in range(1, len(left_contractions) - 1)
        )
        if n_jobs > 1
        else [
            _countPhiTensorKronecker(left_contractions[i], right_contractions[i])
            for i in range(1, len(left_contractions) - 1)
        ]
    )
