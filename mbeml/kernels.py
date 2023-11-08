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


class Masking(sk_kernels.Kernel):
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

    def diag(self, X):
        return self.kernel.diag(X[:, self.mask])

    def __repr__(self):
        return "Masking({0})".format(self.kernel)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()


class TwoBodyKernel(sk_kernels.Kernel):
    def __init__(self, kernel, n_core_features=7):
        self.kernel = kernel
        self.n_core_features = n_core_features

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
        # Build a (n * 6) x (n_core_features + n_lig_features) array
        n = X.shape[0]
        X_reshaped = np.concatenate(
            [
                np.tile(X[:, np.newaxis, : self.n_core_features], (1, 6, 1)),
                X[:, self.n_core_features :].reshape(n, 6, -1),
            ],
            axis=-1,
        ).reshape(n * 6, -1)

        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            # Call kernel using X_reshaped
            K, K_grad = self.kernel(X_reshaped, eval_gradient=True)
            # Reshape and sum over the 6 ligands
            return (
                K.reshape(n, 6, n, 6).sum(axis=(1, 3)) / 36,
                K_grad.reshape(n, 6, n, 6, -1).sum(axis=(1, 3)) / 36,
            )
        if Y is None:
            # Call kernel using X_reshaped, reshape, and sum over the 6 ligands
            return self.kernel(X_reshaped).reshape(n, 6, n, 6).sum(axis=(1, 3)) / 36

        # Since Y is not none:
        # Build a (m * 6) x (n_core_features + n_lig_features) array
        m = Y.shape[0]
        Y_reshaped = np.concatenate(
            [
                np.tile(Y[:, np.newaxis, : self.n_core_features], (1, 6, 1)),
                Y[:, self.n_core_features :].reshape(m, 6, -1),
            ],
            axis=-1,
        ).reshape(m * 6, -1)
        # Call kernel using X_reshaped and Y_reshaped, reshape,
        # and sum over the 6 ligands
        return (
            self.kernel(X_reshaped, Y_reshaped).reshape(n, 6, m, 6).sum(axis=(1, 3))
            / 36
        )

    def diag(self, X):
        return np.diag(self(X))

    def __repr__(self):
        return "TwoBody({0})".format(self.kernel)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()


class TwoBodyKernelNaive(sk_kernels.Kernel):
    def __init__(self, kernel, n_core_features=7):
        self.kernel = kernel
        self.n_core_features = n_core_features

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
            raise NotImplementedError(
                "The TwoBodyKernelNaive implementation does not support gradients. "
                "Use TwoBodyKernel instead."
            )

        n_features = X.shape[1]
        n_lig_features = (n_features - self.n_core_features) // 6
        if Y is None:
            Y = X

        K = np.zeros([X.shape[0], Y.shape[0]])
        for i in range(6):
            # Using masks to ensure we are only working on views instead of copies
            # of the arrays
            mask_i = np.zeros(n_features, dtype=bool)
            mask_i[: self.n_core_features] = True
            mask_i[
                self.n_core_features
                + i * n_lig_features : self.n_core_features
                + (i + 1) * n_lig_features
            ] = True
            for j in range(6):
                mask_j = np.zeros(n_features, dtype=bool)
                mask_j[: self.n_core_features] = True
                mask_j[
                    self.n_core_features
                    + j * n_lig_features : self.n_core_features
                    + (j + 1) * n_lig_features
                ] = True
                K += self.kernel(X[:, mask_i], Y[:, mask_j])
        return K / 36

    def diag(self, X):
        return np.diag(self(X))

    def __repr__(self):
        return "TwoBody({0})".format(self.kernel)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()


class ThreeBodyKernel(sk_kernels.Kernel):
    def __init__(self, kernel, pairs, n_core_features=7):
        self.kernel = kernel
        self.pairs = pairs
        self.n_core_features = n_core_features

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
        n_pairs = len(self.pairs)
        # Build a (n * n_pairs) x (n_core_features + 2 * n_lig_features) array
        n = X.shape[0]
        # Repeat the core featurization
        X_core = np.tile(X[:, np.newaxis, : self.n_core_features], (1, n_pairs, 1))
        X_ligs = X[:, self.n_core_features :].reshape(n, 6, -1)
        # Symmetrize the ligand featurization
        # Slicing using a list, i.e, [i] to keep the dimensions
        X_summed = np.concatenate(
            [0.5 * (X_ligs[:, [i], :] + X_ligs[:, [j], :]) for i, j in self.pairs],
            axis=1,
        )
        X_abs_diff = np.concatenate(
            [0.5 * abs(X_ligs[:, [i], :] - X_ligs[:, [j], :]) for i, j in self.pairs],
            axis=1,
        )
        X_reshaped = np.concatenate([X_core, X_summed, X_abs_diff], axis=-1).reshape(
            n * n_pairs, -1
        )

        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            # Call kernel using X_reshaped
            K, K_grad = self.kernel(X_reshaped, eval_gradient=True)
            # Reshape and sum over the pairs
            return (
                K.reshape(n, n_pairs, n, n_pairs).sum(axis=(1, 3)) / n_pairs**2,
                K_grad.reshape(n, n_pairs, n, n_pairs, -1).sum(axis=(1, 3))
                / n_pairs**2,
            )
        if Y is None:
            # Call kernel using X_reshaped, reshape, and sum over the pairs
            return (
                self.kernel(X_reshaped).reshape(n, n_pairs, n, n_pairs).sum(axis=(1, 3))
                / n_pairs**2
            )

        # Since Y is not None:
        # Build a (m * n_pairs) x (n_core_features + 2 * n_lig_features) array
        m = Y.shape[0]
        Y_core = np.tile(Y[:, np.newaxis, : self.n_core_features], (1, n_pairs, 1))
        Y_ligs = Y[:, self.n_core_features :].reshape(m, 6, -1)
        Y_summed = np.concatenate(
            [0.5 * (Y_ligs[:, [i], :] + Y_ligs[:, [j], :]) for i, j in self.pairs],
            axis=1,
        )
        Y_abs_diff = np.concatenate(
            [0.5 * abs(Y_ligs[:, [i], :] - Y_ligs[:, [j], :]) for i, j in self.pairs],
            axis=1,
        )
        Y_reshaped = np.concatenate([Y_core, Y_summed, Y_abs_diff], axis=-1).reshape(
            m * n_pairs, -1
        )
        # Call kernel using X_reshaped and Y_reshaped, reshape,
        # and sum over the pairs
        return (
            self.kernel(X_reshaped, Y_reshaped)
            .reshape(n, n_pairs, m, n_pairs)
            .sum(axis=(1, 3))
            / n_pairs**2
        )

    def diag(self, X):
        return np.diag(self(X))

    def __repr__(self):
        return "ThreeBody({0})".format(self.kernel)

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return self.kernel.is_stationary()
