import numpy as np
import tt
import typing


def create_exponential_grid(d, n, A, eps=1e-10, alpha=1):
    x = np.reshape(np.linspace(-A, A, n, endpoint=True), (1, n, 1))
    ones = np.reshape(np.ones(n), (1, n, 1))
    base = [ones for _ in range(d)]
    X = []
    for i in range(d):
        current = base.copy()
        current[i] = x.copy()
        X.append(tt.vector.from_list(current))
    c = tt.multifuncrs2(X, lambda x: np.exp(-alpha * np.sqrt(np.sum(x * x, axis=1))), eps=eps, verb=0)
    return tt.vector.to_list(c)


def create_newtonial_potential_grid(d, n, A, eps=1e-10):
    x = np.reshape(np.linspace(-A, A, n, endpoint=True), (1, n, 1))
    ones = np.reshape(np.ones(n), (1, n, 1))
    base = [ones for _ in range(d)]
    X = []
    for i in range(d):
        current = base.copy()
        current[i] = x.copy()
        X.append(tt.vector.from_list(current))
    if d != 2:
        c = tt.multifuncrs2(X, lambda x: np.power(np.sum(x * x, axis=1), (2 - d) / 2), eps=eps, verb=0)
    else:
        c = tt.multifuncrs2(X, lambda x: np.log(np.sum(x * x, axis=1)), eps=eps, verb=0)
    return tt.vector.to_list(c)


def interpolateTTTensor(tt_tensors: typing.List[np.ndarray]):
    def interpolateKernel(kernel):
        answer = np.zeros((kernel.shape[0], 2 * kernel.shape[1] - 1, kernel.shape[2]))
        for i in range(kernel.shape[1]):
            answer[:, 2 * i, :] = kernel[:, i, :]
            if i > 0:
                answer[:, 2 * i - 1, :] = (kernel[:, i, :] + kernel[:, i - 1, :]) / 2
        return answer

    answer = []
    for kernel in tt_tensors:
        answer.append(interpolateKernel(kernel))
    return answer


def compressTTTensor(tt_tensors: typing.List[np.ndarray]):
    def compressKernel(kernel):
        assert kernel.shape[1] % 2 == 1
        new_shape = (kernel.shape[1] + 1) // 2
        answer = np.zeros((kernel.shape[0], new_shape, kernel.shape[2]))
        for i in range(new_shape):
            answer[:, i, :] = kernel[:, 2 * i, :]
        return answer

    answer = []
    for kernel in tt_tensors:
        answer.append(compressKernel(kernel))
    return answer
