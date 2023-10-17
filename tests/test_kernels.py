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
