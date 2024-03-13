import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


__all__ = ["mean_absolute_error", "r2_score"]


def mean_negative_log_likelihood(y_true, y_mean, y_std):
    y_true, y_mean, y_std = y_true.flatten(), y_mean.flatten(), y_std.flatten()
    return np.mean(
        ((y_true - y_mean) ** 2) / (2 * y_std**2)
        - np.log(1 / np.sqrt(2 * np.pi * y_std**2))
    )
