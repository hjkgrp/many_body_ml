import pytest
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from mbeml.kernels import TwoBodyKernel, TwoBodyKernelNaive


@pytest.fixture
def test_data():
    rng = np.random.default_rng(0)
    X = rng.uniform(low=-1.0, high=1.0, size=(5, 7 + 6 * 33))
    Y = rng.uniform(low=-1.0, high=1.0, size=(3, 7 + 6 * 33))
    return X, Y


def test_two_body_kernel(test_data):
    X, Y = test_data
    kernel = TwoBodyKernel(RBF())
    kernel_naive = TwoBodyKernelNaive(RBF())

    np.testing.assert_allclose(kernel(X), kernel_naive(X))
    np.testing.assert_allclose(kernel(X, Y), kernel_naive(X, Y))


def test_two_body_kernel_gradient(test_data):
    X, _ = test_data
    kernel = TwoBodyKernel(RBF())
    _, K_grad = kernel(X, eval_gradient=True)
    kernel_naive = TwoBodyKernelNaive(RBF())
    l0 = np.exp(kernel.theta)
    delta_l = 1e-4

    kernel_naive.theta = np.log(l0 + 0.5 * delta_l)
    K_plus = kernel_naive(X)
    kernel_naive.theta = np.log(l0 - 0.5 * delta_l)
    K_minus = kernel_naive(X)
    kernel_naive.theta = np.log(l0)

    K_grad_num = (K_plus - K_minus) / delta_l
    np.testing.assert_allclose(K_grad_num, K_grad.squeeze(), atol=1e-6)


def test_two_body_kernel_diag(test_data):
    X, _ = test_data
    kernel = TwoBodyKernel(RBF())
    np.testing.assert_allclose(kernel.diag(X), np.diag(kernel(X)))


def test_two_body_permutation(test_data):
    """Checks that the order of ligands does not matter"""
    X, Y = test_data

    kernel = TwoBodyKernel(RBF())
    K = kernel(X, Y)

    X_perm = X.copy()
    Y_perm = Y.copy()
    # Swap ligands 2 and 4
    X_perm[:, 7 + 33 * 1 : 7 + 33 * 2], X_perm[:, 7 + 33 * 3 : 7 + 33 * 4] = (
        X[:, 7 + 33 * 3 : 7 + 33 * 4],
        X[:, 7 + 33 * 1 : 7 + 33 * 2],
    )
    # Swap ligand 3 and 5
    Y_perm[:, 7 + 33 * 2 : 7 + 33 * 3], Y_perm[:, 7 + 33 * 4 : 7 + 33 * 5] = (
        Y[:, 7 + 33 * 4 : 7 + 33 * 5],
        Y[:, 7 + 33 * 2 : 7 + 33 * 3],
    )
    K_perm = kernel(X_perm, Y_perm)
    np.testing.assert_allclose(K_perm, K, atol=1e-6)


def test_two_body_additivity(atol=1e-6):
    """Checks that additivity is perfectly respected"""
    rng = np.random.default_rng(0)
    core = rng.uniform(low=-1.0, high=1.0, size=(1, 7))
    lig_A = rng.uniform(low=-1.0, high=1.0, size=(1, 33))
    lig_B = rng.uniform(low=-1.0, high=1.0, size=(1, 33))

    X_homoleptic_A = np.concatenate([core] + [lig_A] * 6, axis=-1)
    X_homoleptic_B = np.concatenate([core] + [lig_B] * 6, axis=-1)

    X_interp = []
    for i in range(1, 6):
        X_interp.append(
            np.concatenate([core] + [lig_A] * (6 - i) + [lig_B] * i, axis=-1)
        )

    X = np.concatenate(
        [X_homoleptic_A] + X_interp + [X_homoleptic_B],
        axis=0,
    )

    kernel = TwoBodyKernel(RBF())
    K = kernel(X)
    # Vector of kernel similarities to X_homoleptic_A
    k_vec_A = K[-7, :]
    k_vec_B = K[-1, :]

    # Check that they add up to a constant value
    np.testing.assert_allclose(
        np.diff(k_vec_A + k_vec_B), np.zeros(len(k_vec_A) - 1), atol=atol
    )
    # Check monotonic increase / decrease
    np.testing.assert_allclose(
        k_vec_A, k_vec_A[0] + np.arange(7) * (k_vec_A[-1] - k_vec_A[0]) / 6
    )
    np.testing.assert_allclose(
        k_vec_B, k_vec_B[0] + np.arange(7) * (k_vec_B[-1] - k_vec_B[0]) / 6
    )
