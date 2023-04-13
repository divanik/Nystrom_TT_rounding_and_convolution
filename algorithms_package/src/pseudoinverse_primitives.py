import typing
import numpy as np
from numpy.linalg import lstsq

from algorithms_package.src import primitives


def processTensorsTakeLeft(psi_tensors: typing.List[np.array], phi_tensors: typing.List[np.array]):
    answer = []
    modes = primitives.countModes(psi_tensors)
    for i in range(0, len(modes) - 1):
        R = np.swapaxes(psi_tensors[i], 0, 2).reshape((psi_tensors[i].shape[2], -1), order='F')
        new_shape = (phi_tensors[i].shape[0], psi_tensors[i].shape[1], psi_tensors[i].shape[0])
        g = lstsq(phi_tensors[i].T, R, rcond=1e-8)
        print(g[1], g[2], g[3])
        p = g[0].reshape(new_shape, order='F')
        answer.append(np.swapaxes(p, 0, 2))
    answer.append(psi_tensors[-1])
    return answer


def processTensorsTakeRight(psi_tensors: typing.List[np.array], phi_tensors: typing.List[np.array]):
    answer = [psi_tensors[0]]
    modes = primitives.countModes(psi_tensors)
    for i in range(1, len(modes)):
        R = psi_tensors[i].reshape((psi_tensors[i].shape[0], -1), order='F')
        new_shape = (phi_tensors[i - 1].shape[1], psi_tensors[i].shape[1], psi_tensors[i].shape[2])
        answer.append(lstsq(phi_tensors[i - 1], R, rcond=1e-10)[0].reshape(new_shape, order='F'))
    return answer
