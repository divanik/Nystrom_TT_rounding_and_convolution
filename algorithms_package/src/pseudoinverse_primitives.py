import typing
import numpy as np
from numpy.linalg import lstsq
from time import time

from algorithms_package.src import primitives
from joblib import Parallel, delayed


def processPhiTensorFromRight(psi_tensor, phi_tensor):
    R = np.swapaxes(psi_tensor, 0, 2).reshape((psi_tensor.shape[2], -1), order='F')
    new_shape = (phi_tensor.shape[0], psi_tensor.shape[1], psi_tensor.shape[0])
    g = lstsq(phi_tensor.T, R, rcond=1e-10)
    p = g[0].reshape(new_shape, order='F')
    return np.swapaxes(p, 0, 2)


def processPhiTensorFromLeft(psi_tensor, phi_tensor):
    R = psi_tensor.reshape((psi_tensor.shape[0], -1), order='F')
    new_shape = (phi_tensor.shape[1], psi_tensor.shape[1], psi_tensor.shape[2])
    return lstsq(phi_tensor, R, rcond=1e-10)[0].reshape(new_shape, order='F')


def processTensorsTakeLeft(psi_tensors: typing.List[np.array], phi_tensors: typing.List[np.array]):
    modes_size = len(primitives.countModes(psi_tensors))
    answer = []
    times = []
    for i in range(modes_size - 1):
        time0 = time()
        answer.append(processPhiTensorFromRight(psi_tensors[i], phi_tensors[i]))
        times.append(time() - time0)
    answer.append(psi_tensors[-1])
    times.append(0)
    return answer, times


def processTensorsTakeRight(psi_tensors: typing.List[np.array], phi_tensors: typing.List[np.array]):
    modes_size = len(primitives.countModes(psi_tensors))
    answer = [psi_tensors[0]]
    times = [0]
    for i in range(1, modes_size):
        time0 = time()
        answer.append(processPhiTensorFromLeft(psi_tensors[i], phi_tensors[i]))
        times.append(time() - time0)
    return answer, times
