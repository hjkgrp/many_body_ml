import numpy as np
from scipy.stats import norm
from mbeml.metrics import mean_negative_log_likelihood


def naive_nll(y_true, y_mean, y_std):
    nll = np.zeros(len(y_true))
    for i, (true_mean, pred_mean, pred_std) in enumerate(zip(y_true, y_mean, y_std)):
        nll[i] = -norm.logpdf(true_mean, loc=pred_mean, scale=pred_std)
    return np.mean(nll)


def test_mean_negative_log_likelihood():
    y_true = np.array([6.4, 1.0, -1.2, 40.2])
    y_mean = np.array([8.2, -0.2, -0.9, 38.4])
    y_std = np.array([1.2, 0.8, 1.1, 1.4])

    ref = naive_nll(y_true, y_mean, y_std)
    mnll = mean_negative_log_likelihood(y_true, y_mean, y_std)

    np.testing.assert_allclose(mnll, ref)


def test_mean_negative_log_likelihood_reshaped():
    y_true = np.array([6.4, 1.0, -1.2, 40.2]).reshape(-1, 1)
    y_mean = np.array([8.2, -0.2, -0.9, 38.4]).reshape(-1, 1)
    y_std = np.array([1.2, 0.8, 1.1, 1.4]).reshape(-1, 1)

    ref = naive_nll(y_true, y_mean, y_std)
    mnll = mean_negative_log_likelihood(y_true, y_mean, y_std)

    np.testing.assert_allclose(mnll, ref)


def test_mean_negative_log_likelihood_mixed_shape():
    y_true = np.array([6.4, 1.0, -1.2, 40.2]).reshape(-1, 1)
    y_mean = np.array([8.2, -0.2, -0.9, 38.4])
    y_std = np.array([1.2, 0.8, 1.1, 1.4])

    ref = naive_nll(y_true, y_mean, y_std)
    mnll = mean_negative_log_likelihood(y_true, y_mean, y_std)

    np.testing.assert_allclose(mnll, ref)
