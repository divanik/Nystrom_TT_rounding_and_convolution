import typing
import numpy as np


def createRandomTensor(modes: typing.List[np.int32], ranks: typing.List[np.int32], seed: int):
    np.random.seed(seed)
    answer = []
    for idx in range(len(modes)):
        l1 = 1 if idx == 0 else ranks[idx - 1]
        l2 = 1 if idx == len(modes) - 1 else ranks[idx]
        tensor = np.random.normal(loc=0.0, scale=1 / (np.sqrt(l1 * modes[idx] * l2)), size=(l1, modes[idx], l2))
        answer.append(tensor)
    return answer


def createExampleTensor(modes: typing.List[np.int32], ranks: typing.List[np.int32], variance: float, seed: int):
    np.random.seed(seed)
    answer = []
    for idx in range(len(modes)):
        l1 = 1 if idx == 0 else ranks[idx - 1]
        l2 = 1 if idx == len(modes) - 1 else ranks[idx]
        tensor = np.random.normal(loc=0.0, scale=variance, size=(l1, modes[idx], l2))
        answer.append(tensor)
    return answer


def createUnbiasedNormalTensor(modes: typing.List[np.int32], ranks: typing.List[np.int32], seed: int):
    np.random.seed(seed)
    answer = []
    for idx in range(len(modes)):
        l1 = 1 if idx == 0 else ranks[idx - 1]
        l2 = 1 if idx == len(modes) - 1 else ranks[idx]
        tensor = np.random.normal(loc=0.0, scale=1 / np.sqrt(np.sqrt(l1 * l2)), size=(l1, modes[idx], l2))
        answer.append(tensor)
    return answer


def createUnbiasedRademacherTensor(modes: typing.List[np.int32], ranks: typing.List[np.int32], seed: int):
    np.random.seed(seed)
    answer = []
    for idx in range(len(modes)):
        l1 = 1 if idx == 0 else ranks[idx - 1]
        l2 = 1 if idx == len(modes) - 1 else ranks[idx]
        tensor = (1 - 2 * np.random.randint(2, size=(l1, modes[idx], l2))) / np.sqrt(np.sqrt(l1 * l2))
        answer.append(tensor)
    return answer


def createPositiveExampleTensor(modes: typing.List[np.int32], ranks: typing.List[np.int32], variance: float, seed: int):
    np.random.seed(seed)
    answer = []
    for idx in range(len(modes)):
        l1 = 1 if idx == 0 else ranks[idx - 1]
        l2 = 1 if idx == len(modes) - 1 else ranks[idx]
        tensor = np.abs(np.random.normal(loc=0.0, scale=variance, size=(l1, modes[idx], l2)))
        answer.append(tensor)
    return answer
