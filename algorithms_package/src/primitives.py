import typing
import numpy as np
import logging


def countTensor(tt_tensors: typing.List[np.array]):
    answer = np.ones(1)
    for tensor in tt_tensors:
        answer = np.einsum('...i,ijk->...jk', answer, tensor)
    return np.einsum('...i,i->...', answer, np.ones(1))


def frob(tensor: np.array):
    return np.sqrt(np.sum(tensor * tensor))


def makeHorizontalUnfolding(tensor: np.array):
    return np.reshape(np.einsum('ijk->ikj', tensor), (tensor.shape[0], -1), order='F')


def fromHorizontalUnfolding(matrix: np.array, shape: typing.Tuple[int]):
    good_shape = (shape[0], shape[2], shape[1])
    return np.einsum('ijk->ikj', np.reshape(matrix, good_shape, order='F'))


def makeVerticalUnfolding(tensor: np.array):
    return np.reshape(tensor, (-1, tensor.shape[2]), order='F')


def fromVerticalUnfolding(matrix: np.array, shape: typing.Tuple[int]):
    return np.reshape(matrix, shape, order='F')


def isTTformat(tensor: typing.List[np.ndarray]):
    for p in tensor:
        if len(p.shape) != 3:
            return False
    if tensor[0].shape[0] != 1 or tensor[len(tensor) - 1].shape[2] != 1:
        return False
    for i in range(len(tensor) - 1):
        if tensor[i].shape[2] != tensor[i + 1].shape[0]:
            return False
    return True


def isEqualTTForms(first: typing.List[np.ndarray], second: typing.List[np.ndarray]):
    if not isTTformat(first) or not isTTformat(second):
        return False
    if len(first) != len(second):
        return False
    for i in range(len(first) - 1):
        if first[i].shape[1] != second[i].shape[1]:
            return False
    return True


def scalarProduct(first: typing.List[np.ndarray], second: typing.List[np.ndarray]):
    current_tensor = np.einsum('ijk, ljm -> ilkm', first[0], second[0])
    for i in range(1, len(first)):
        current_tensor = np.einsum('ijkl,kmn->ijlmn', current_tensor, first[i])
        current_tensor = np.einsum(
            'ijlmn,lmp->ijnp', current_tensor, second[i])
    return current_tensor[0, 0, 0, 0]


def countFrobeniusDistance(first: typing.List[np.ndarray], second: typing.List[np.ndarray]):
    return np.sqrt(scalarProduct(first, first) + scalarProduct(second, second) - 2 * scalarProduct(first, second))


def countFrobeniusNorm(first: typing.List[np.ndarray]):
    return np.sqrt(scalarProduct(first, first))


def countFrobeniusDistanceSlow(first: typing.List[np.ndarray], second: typing.List[np.ndarray]):
    return frob(countTensor(first) - countTensor(second))


def formSumTensor(first: typing.List[np.ndarray], second: typing.List[np.ndarray]):
    answer = np.array((first.shape[0] + second.shape[0],
                      first.shape[1], first.shape[2] + second.shape[2]))
    answer[0:first.shape[0], :, 0: first.shape[2]] = first
    answer[-second.shape[0] - 1:-1, :, -second.shape[2] - 1:-1] = second
    return answer


def directSum(first: typing.List[np.ndarray], second: typing.List[np.ndarray]):
    answer = []
    for i in range(len(first)):
        answer.append(formSumTensor(first[i], second[i]))
    return answer


def tensorsRelativeComparance(first: np.ndarray, second: np.ndarray):
    return np.linalg.norm((first - second).flatten()) / np.sqrt(np.linalg.norm(first.flatten()) * np.linalg.norm(second.flatten()))


def ttTensorsRelativeComparance(first: typing.List[np.ndarray], second: typing.List[np.ndarray]):
    sp1 = scalarProduct(first, first)
    sp2 = scalarProduct(second, second)
    sp = scalarProduct(first, second)
    return np.sqrt((sp1 + sp2 - 2 * sp) / (np.sqrt(sp1) * np.sqrt(sp2)))


def ttTensorsUnsymmetricalRelativeComparance(result: typing.List[np.ndarray], reference: typing.List[np.ndarray]):
    sp1 = scalarProduct(result, result)
    sp2 = scalarProduct(reference, reference)
    sp = scalarProduct(result, reference)
    return np.sqrt(np.abs(sp1 + sp2 - 2 * sp) / sp2)


def countModes(tt_tensors: typing.List[np.ndarray]):
    modes = []
    for tensor in tt_tensors:
        modes.append(tensor.shape[1])
    return modes


def twoSidedPaddingTTTensor(tt_tensors: typing.List[np.ndarray], padding: typing.List[typing.Tuple[int]]):
    assert len(tt_tensors) == len(padding)
    answer = []
    for i in range(len(tt_tensors)):
        kernel = np.zeros((tt_tensors[i].shape[0], padding[i][0] +
                          tt_tensors[i].shape[1] + padding[i][1], tt_tensors[i].shape[2]))
        kernel[:, padding[i][0]:-padding[i][1], :] = tt_tensors[i]
        answer.append(kernel)
    return answer


def twoSidedCuttingTTTensor(tt_tensors: typing.List[np.ndarray], cutting: typing.List[typing.Tuple[int]]):
    assert len(tt_tensors) == len(cutting)
    answer = []
    for i in range(len(tt_tensors)):
        kernel = tt_tensors[i][:, cutting[i][0]: -cutting[i][1], :]
        answer.append(kernel)
    return answer

