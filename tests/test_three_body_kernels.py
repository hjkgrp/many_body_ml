import numpy as np
from sklearn.gaussian_process.kernels import RBF
from mbeml.kernels import ThreeBodyKernel
from mbeml.constants import cis_pairs, trans_pairs


def test_three_body_interpolation():
    rng = np.random.default_rng(0)
    core = rng.uniform(low=-1.0, high=1.0, size=(1, 7))
    lig_A = rng.uniform(low=-1.0, high=1.0, size=(1, 33))
    lig_B = rng.uniform(low=-1.0, high=1.0, size=(1, 33))

    X_homoleptic_A = np.concatenate([core] + [lig_A] * 6, axis=-1)
    X_homoleptic_B = np.concatenate([core] + [lig_B] * 6, axis=-1)

    X_5A_1B = np.concatenate([core] + [lig_A] * 5 + [lig_B], axis=-1)
    X_4A_2B_cis = np.concatenate([core] + [lig_A] * 3 + [lig_B] * 2 + [lig_A], axis=-1)
    X_4A_2B_trans = np.concatenate([core] + [lig_A] * 4 + [lig_B] * 2, axis=-1)
    X_3A_3B_fac = np.concatenate([core] + [lig_A] * 2 + [lig_B] * 3 + [lig_A], axis=-1)
    X_3A_3B_mer = np.concatenate([core] + [lig_A] * 3 + [lig_B] * 3, axis=-1)
    X_2A_4B_cis = np.concatenate([core] + [lig_A] * 2 + [lig_B] * 4, axis=-1)
    X_2A_4B_trans = np.concatenate(
        [core] + [lig_A, lig_B, lig_A] + [lig_B] * 3, axis=-1
    )
    X_1A_5B = np.concatenate([core] + [lig_A] + [lig_B] * 5, axis=-1)

    X = np.concatenate(
        [
            X_homoleptic_A,
            X_homoleptic_B,
            X_3A_3B_fac,
            X_3A_3B_mer,
        ],
        axis=0,
    )

    Y = np.concatenate(
        [
            X_5A_1B,
            X_4A_2B_cis,
            X_4A_2B_trans,
            X_2A_4B_trans,
            X_2A_4B_cis,
            X_1A_5B,
        ],
        axis=0,
    )

    kernel = ThreeBodyKernel(RBF(), pairs=cis_pairs) + ThreeBodyKernel(
        RBF(), pairs=trans_pairs
    )
    K = kernel(X, Y)
    # Contributions of the homoleptics should be complementary
    np.testing.assert_allclose(K[0, :], K[1, ::-1])
    # Contributions of fac/mer should be "symmetric"
    np.testing.assert_allclose(K[2, :3], K[2, -3:][::-1])
    np.testing.assert_allclose(K[3, :3], K[3, -3:][::-1])
