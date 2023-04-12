import typing
import numpy as np
from algorithms_package.src import primitives


def orthogonalizeRL(tt_tensors: typing.List[np.array]):
    answer = tt_tensors.copy()
    for idx in range(len(tt_tensors) - 1, 0, -1):
        tensor = answer[idx]
        y = primitives.makeHorizontalUnfolding(tensor)
        y, r = np.linalg.qr(y.T)
        initial_shape = (y.shape[1], tensor.shape[1], tensor.shape[2])
        answer[idx] = primitives.fromHorizontalUnfolding(y.T, initial_shape)
        answer[idx - 1] = np.einsum('ijk,kl->ijl', answer[idx - 1], r.T)
    return answer
