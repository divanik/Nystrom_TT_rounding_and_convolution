import typing
import logging
import numpy as np
from algorithms_package.src import primitives

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
        p = cronMulVecR(tt1L, tt1R, answer[-1])
        answer.append(np.einsum('udl,abdu->abl', tt2, p))
    return answer
