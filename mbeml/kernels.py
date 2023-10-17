import numpy as np
from sklearn.gaussian_process import kernels as sk_kernels


class Normalization(sk_kernels.NormalizedKernelMixin, sk_kernels.Kernel):
    def __init__(self, kernel):
        self.kernel = kernel

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) is not type(b):
            return False
        return self.kernel == b.kernel

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Kxy, Kxy_gradient = self.kernel(X, eval_gradient=True)
            Kxx = np.diag(Kxy)
            Kxx_sqrt = np.sqrt(Kxx)
            Kxx_gradient = Kxy_gradient.diagonal().T
            grad = (
                2 * np.outer(Kxx, Kxx)[:, :, np.newaxis] * Kxy_gradient
                - Kxy[:, :, np.newaxis]
                * (
                    Kxx_gradient[:, np.newaxis, :] * Kxx[np.newaxis, :, np.newaxis]
                    + Kxx_gradient[np.newaxis, :, :] * Kxx[:, np.newaxis, np.newaxis]
                )
            ) / (2 * np.outer(Kxx_sqrt, Kxx_sqrt) ** 3)[:, :, np.newaxis]
            return Kxy / np.outer(Kxx_sqrt, Kxx_sqrt), grad
        else:
            if Y is None:
                Kxy = self.kernel(X, eval_gradient=False)
                Kxx = np.diag(Kxy)
                return Kxy / np.sqrt(np.outer(Kxx, Kxx))
            else:
                Kxy = self.kernel(X, Y, eval_gradient=False)
                Kxx = self.kernel.diag(X)
                Kyy = self.kernel.diag(Y)
                return Kxy / np.sqrt(np.outer(Kxx, Kyy))

    def __repr__(self):
        return "Normalized({0})".format(self.kernel)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()


class Masking(sk_kernels.NormalizedKernelMixin, sk_kernels.Kernel):
    def __init__(self, mask, kernel):
        self.mask = mask
        self.kernel = kernel

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            return self.kernel(X[:, self.mask], Y=None, eval_gradient=eval_gradient)
        return self.kernel(
            X[:, self.mask], Y=Y[:, self.mask], eval_gradient=eval_gradient
        )

    def __repr__(self):
        return "Masking({0})".format(self.kernel)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()


class TwoBodyKernel(sk_kernels.Kernel):
    def __init__(self, kernel):
        self.kernel = kernel

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) is not type(b):
            return False
        return self.kernel == b.kernel

    def __call__(self, X, Y=None, eval_gradient=False):
        n = X.shape[0]

        X_reshaped = np.concatenate(
            [np.tile(X[:, np.newaxis, :7], (1, 6, 1)), X[:, 7:].reshape(-1, 6, 33)],
            axis=-1,
        ).reshape(6 * n, -1)

        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            K, K_grad = self.kernel(X_reshaped, eval_gradient=True)
            return K.reshape(n, 6, n, 6).sum(axis=(1, 3)), K_grad.reshape(
                n, 6, n, 6, -1
            ).sum(axis=(1, 3))
        if Y is None:
            return self.kernel(X_reshaped).reshape(n, 6, n, 6).sum(axis=(1, 3))
        m = Y.shape[0]
        Y_reshaped = np.concatenate(
            [np.tile(Y[:, np.newaxis, :7], (1, 6, 1)), Y[:, 7:].reshape(-1, 6, 33)],
            axis=-1,
        ).reshape(6 * m, -1)
        return self.kernel(X_reshaped, Y_reshaped).reshape(n, 6, m, 6).sum(axis=(1, 3))

    def diag(self, X):
        return np.diag(self(X))

    def __repr__(self):
        return "TwoBody({0})".format(self.kernel)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()


class TwoBodyKernelNaive(sk_kernels.Kernel):
    def __init__(self, kernel):
        self.kernel = kernel

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        self.kernel.theta = theta

    @property
    def bounds(self):
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) is not type(b):
            return False
        return self.kernel == b.kernel

    def __call__(self, X, Y=None, eval_gradient=False):
        n_features = X.shape[1]
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        K = np.zeros([X.shape[0], Y.shape[0]])
        for i in range(6):
            # Using masks to ensure we are only working on views instead of copies
            # of the arrays
            mask_i = np.zeros(n_features, dtype=bool)
            mask_i[:7] = True
            mask_i[7 + i * 33 : 7 + (i + 1) * 33] = True
            for j in range(6):
                mask_j = np.zeros(n_features, dtype=bool)
                mask_j[:7] = True
                mask_j[7 + j * 33 : 7 + (j + 1) * 33] = True
                K += self.kernel(X[:, mask_i], Y[:, mask_j])
        if eval_gradient:
            K_gradient = np.empty((X.shape[0], X.shape[0], 0))
            return K, K_gradient
        else:
            return K

    def diag(self, X):
        return np.diag(self(X))

    def __repr__(self):
        return "TwoBody({0})".format(self.kernel)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()
