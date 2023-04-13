import typing
import numpy as np
from algorithms_package.src import (
    contraction,
    primitives,
    random_tensor_generation,
    pseudoinverse_primitives,
)


# def ttRoundingWithRanks(tt_tensors: typing.List[np.array], desired_ranks: typing.List[int]):
#     answer = orthogonalize.orthogonalizeRL(tt_tensors)
#     for idx in range(len(answer) - 1):
#         tensor = answer[idx]
#         shape = tensor.shape
#         y = primitives.makeVerticalUnfolding(tensor)
#         y, r = np.linalg.qr(y)
#         shape = (tensor.shape[0], tensor.shape[1], y.shape[1])
#         answer[idx] = primitives.fromVerticalUnfolding(y, shape)
#         U, S, Vt = np.linalg.svd(r, full_matrices=False)
#         l = desired_ranks[idx]
#         if l < S.size:
#             U = U[:, :l]
#             S = S[:l]
#             Vt = Vt[:l, :]
#         answer[idx] = np.einsum('ijk,kl->ijl', answer[idx], U)

#         SVt = np.diag(S) @ Vt
#         answer[idx + 1] = np.einsum('ij,jkl->ikl', SVt, answer[idx + 1])
#     return answer


# def orthogonalizeThenRandomize(tt_tensors: typing.List[np.array], desired_ranks: typing.List[int], seed: int):
#     answer = orthogonalize.orthogonalizeRL(tt_tensors)
#     np.random.seed(seed)
#     for idx in range(len(answer) - 1):
#         tensor = answer[idx]
#         shape = tensor.shape
#         z = primitives.makeVerticalUnfolding(tensor)
#         omega = np.random.normal(
#             loc=0.0, scale=1 / (z.shape[1] * desired_ranks[idx]), size=(z.shape[1], desired_ranks[idx])
#         )
#         y = z @ omega
#         v, _ = np.linalg.qr(y)
#         answer[idx] = primitives.fromVerticalUnfolding(v, (shape[0], shape[1], v.shape[1]))
#         m = v.T @ z
#         answer[idx + 1] = np.einsum('ij,jkl->ikl', m, answer[idx + 1])
#     return answer


def randomizeThenOrthogonalize(tt_tensors: typing.List[np.array], desired_ranks: typing.List[np.array], seed: int):
    random_tensor = random_tensor_generation.createRandomTensor(primitives.countModes(tt_tensors), desired_ranks, seed)
    contractions = contraction.partialContractionsRL(tt_tensors, random_tensor)
    answer = tt_tensors.copy()
    for idx in range(len(tt_tensors) - 1):
        z = primitives.makeVerticalUnfolding(answer[idx])
        shape = answer[idx].shape
        y = z @ contractions[idx]
        y, _ = np.linalg.qr(y)
        answer[idx] = primitives.fromVerticalUnfolding(y, (shape[0], shape[1], y.shape[1]))
        m = y.T @ z
        answer[idx + 1] = np.einsum('ij, jkl->ikl', m, answer[idx + 1])
    return answer


def twoSidedRounding(
    tt_tensors: typing.List[np.array],
    desired_ranks: typing.List[np.array],
    auxiliary_ranks: typing.List[np.array],
    seed: int,
    leave_left=True,
):
    modes = primitives.countModes(tt_tensors)
    if not leave_left:
        desired_ranks, auxiliary_ranks = auxiliary_ranks, desired_ranks
    left_random_tensor = random_tensor_generation.createRandomTensor(modes, desired_ranks, seed)
    right_random_tensor = random_tensor_generation.createRandomTensor(modes, auxiliary_ranks, seed)
    return _twoSidedRounding(tt_tensors, left_random_tensor, right_random_tensor, leave_left)


def _twoSidedRounding(
    tt_tensors: typing.List[np.array],
    left_random_tensor: np.array,
    right_random_tensor: np.array,
    leave_left: bool,
):
    left_contractions = contraction.partialContractionsLR(tt_tensors, left_random_tensor)
    right_contractions = contraction.partialContractionsRL(tt_tensors, right_random_tensor)
    psi_tensors = contraction.countPsiTensors(left_contractions, tt_tensors, right_contractions)
    phi_tensors = contraction.countPhiTensors(left_contractions, right_contractions)
    print("PSI!!")
    print(psi_tensors)
    print()
    print()
    print("PHI!!")
    print(phi_tensors)
    print()
    print()
    print("KEK!!")
    print()
    print()
    for phi_tensor in phi_tensors:
        print(np.linalg.svd(phi_tensor)[1])
    return (
        pseudoinverse_primitives.processTensorsTakeLeft(psi_tensors, phi_tensors)
        if leave_left
        else pseudoinverse_primitives.processTensorsTakeRight(psi_tensors, phi_tensors)
    )
