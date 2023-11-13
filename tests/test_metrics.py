import numpy as np
from scipy.stats import norm
from mbeml.metrics import mean_negative_log_likelihood


def test_mean_negative_log_likelihood():
    y_true = np.array([6.4, 1.0, -1.2, 40.2])
    y_mean = np.array([8.2, -0.2, -0.9, 38.4])
    y_std = np.array([1.2, 0.8, 1.1, 1.4])

    ref = np.mean(-np.log(norm.pdf(y_true, loc=y_mean, scale=y_std)))
    mnll = mean_negative_log_likelihood(y_true, y_mean, y_std)

    np.testing.assert_allclose(mnll, ref)
