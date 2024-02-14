import numpy as np
import matplotlib.pyplot as plt


def generate(n, m, d, delta_mu, no_outlier_s, no_outlier_t, signal):
    mu_Xs = np.ones(d)
    mu_Xt = mu_Xs + delta_mu

    true_Xs = np.array([mu_Xs for _ in range(n)])
    true_Xt = np.array([mu_Xt for _ in range(m)])

    ys = np.zeros(n)
    yt = np.zeros(m)

    # Generate outlier
    idx_s = np.random.randint(n, size=no_outlier_s)
    idx_s = np.unique(idx_s)
    true_Xs[idx_s] = true_Xs[idx_s] + 2
    ys[idx_s] = 1

    idx_t = np.random.randint(m, size=no_outlier_t)
    idx_t = np.unique(idx_t)
    true_Xt[idx_t] = true_Xt[idx_t] + signal
    if signal != 0.0:
        yt[idx_t] = 1

    Xs_obs = true_Xs + np.random.multivariate_normal(np.zeros(d), np.identity(d), n)
    Xt_obs = true_Xt + np.random.multivariate_normal(np.zeros(d), np.identity(d), m)

    return Xs_obs, Xt_obs, true_Xs, true_Xt, ys, yt


