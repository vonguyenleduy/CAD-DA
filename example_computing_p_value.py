import numpy as np
import matplotlib.pyplot as plt
# import statsmodels.api as sm

# import ot

import gen_data
import util
import parametric_si


def find_truncation_interval_sign(list_sign, list_eta_2d_minus_eta_3_d, a, b, n_s, d):
    list_u = []
    list_v = []

    for each_dim in range(d):
        u = - list_sign[each_dim] * np.dot(list_eta_2d_minus_eta_3_d[each_dim].T, b[n_s * d:, :])[0][0]
        v = list_sign[each_dim] * np.dot(list_eta_2d_minus_eta_3_d[each_dim].T, a[n_s * d:, :])[0][0]

        list_u.append(u)
        list_v.append(v)

    Vminus = np.NINF
    Vplus = np.Inf

    # Selection event for Outlier and Sign
    for i in range(len(list_u)):
        a_coef = list_u[i]
        b_coef = list_v[i]

        if a_coef == 0.0:
            if b_coef < 0:
                print('error')

            continue

        temp = b_coef / a_coef

        if a_coef > 0:
            Vplus = min(Vplus, temp)
        elif a_coef < 0:
            Vminus = max(Vminus, temp)

    return Vminus, Vplus


def run():
    n_s = 10
    n_t = 5
    d = 1
    delta_mu = 2
    dim_t = n_s * n_t

    no_outlier_s = 5
    no_outlier_t = 5

    signal = 0

    alpha = 1.5

    threshold = 20

    Xs_obs, Xt_obs, true_Xs, true_Xt, ys, yt = gen_data.generate(n_s, n_t, d, delta_mu, no_outlier_s, no_outlier_t, signal)

    Xs_obs_vec = Xs_obs.flatten().copy().reshape((n_s * d, 1))
    Xt_obs_vec = Xt_obs.flatten().copy().reshape((n_t * d, 1))

    true_Xs_vec = true_Xs.flatten().copy().reshape((n_s * d, 1))
    true_Xt_vec = true_Xt.flatten().copy().reshape((n_t * d, 1))

    data_obs = np.vstack((Xs_obs_vec, Xt_obs_vec)).copy()

    # Cost matrix
    C = np.zeros((n_s, n_t))

    for i in range(n_s):
        e_x = Xs_obs[i, :]
        for j in range(n_t):
            e_y = Xt_obs[j, :]
            C[i, j] = np.sum((e_x - e_y)**2)

    c_vec = C.copy().flatten().reshape((dim_t, 1))

    # LP
    S, u, G, h = util.construct_S_u_G_h(n_s, n_t)
    lp_res = util.LP_solver(c_vec, S, u, G, h)

    # OT plan vector
    t_hat = np.around(lp_res.x, 10)

    T_hat = t_hat.reshape((n_s, n_t))

    # Active set and non-active set
    A = lp_res.basis.copy().tolist()
    Ac = []
    for i in range(dim_t):
        if i not in A:
            Ac.append(i)

    Xs_obs_tilde = n_s * np.dot(T_hat, Xt_obs)

    merge_matrix = np.vstack((Xs_obs_tilde, Xt_obs))

    detect_outlier_idx = []

    for each_dim in range(d):
        merged_data = merge_matrix[:, each_dim]

        median_1 = np.median(merged_data)
        list_abs_deviation = np.abs(merged_data - median_1)
        median_2 = np.median(list_abs_deviation)

        lower = median_1 - alpha * median_2
        upper = median_1 + alpha * median_2

        for j in range(n_t):
            element = Xt_obs[j][each_dim]
            if (element < lower) or (element > upper):
                if j not in detect_outlier_idx:
                    detect_outlier_idx.append(j)

    len_detect_outlier_idx = len(detect_outlier_idx)

    if (len_detect_outlier_idx == 0) or (len_detect_outlier_idx == n_t):
        return None, None

    idx = np.random.randint(len_detect_outlier_idx)
    selected_j = detect_outlier_idx[idx]

    # Construct eta
    eta_1 = np.zeros((n_s * d, 1))

    eta_2 = np.zeros((n_t * d, 1))
    eta_3 = np.zeros((n_t * d, 1))

    list_sign = []
    list_eta_2d_minus_eta_3_d = []

    for each_dim in range(d):
        eta_2d = np.zeros((n_t, 1))
        eta_2d[selected_j] = 1.0

        dim_vector = np.zeros((d, 1))
        dim_vector[each_dim] = 1.0

        eta_2d = np.kron(eta_2d, dim_vector)

        eta_3d = (1 / (n_t - len_detect_outlier_idx)) * np.ones((n_t, 1))
        for element in detect_outlier_idx:
            eta_3d[element] = 0.0

        eta_3d = np.kron(eta_3d, dim_vector)

        sign = np.sign(np.dot((eta_2d - eta_3d).T, Xt_obs_vec)[0][0])

        list_sign.append(sign)
        list_eta_2d_minus_eta_3_d.append(eta_2d - eta_3d)

        eta_2 = eta_2 + sign * eta_2d
        eta_3 = eta_3 + sign * eta_3d

    eta = np.vstack((eta_1, eta_2 - eta_3))

    # Construct etaTdata
    etaTdata = np.dot(eta.T, data_obs)[0][0]

    # Construct a_line and b_line
    a, b = util.compute_a_b(data_obs, eta, n_s * d + n_t * d)

    # parametric_si.run_parametric_si(a, b, n_s, n_t, d, alpha, threshold)

    list_zk, list_outlier_idx = parametric_si.run_parametric_si(a, b, n_s, n_t, d, alpha, threshold)

    Vminus_sign, Vplus_sign = find_truncation_interval_sign(list_sign, list_eta_2d_minus_eta_3_d, a, b, n_s, d)

    z_interval = []

    for i in range(len(list_outlier_idx)):
        if np.array_equal(np.sort(detect_outlier_idx), np.sort(list_outlier_idx[i])):
            z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

    new_z_interval = []

    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    z_interval = new_z_interval

    final_list_interval = []

    for interval in z_interval:
        returned_result = util.intersect([Vminus_sign, Vplus_sign], interval)
        if len(returned_result) != 0:
            final_list_interval.append(returned_result)

    true_data = np.vstack((true_Xs_vec, true_Xt_vec)).copy()
    tn_mu = np.dot(eta.T, true_data)[0][0]
    cov = np.identity(n_s * d + n_t * d)

    # pivot = util.pivot_with_constructed_interval(final_list_interval, eta, etaTdata, cov, tn_mu)
    # return np.around(tn_mu, 5), pivot

    pivot = util.pivot_with_constructed_interval(final_list_interval, eta, etaTdata, cov, 0)
    p_value = 2 * min(pivot, 1 - pivot)

    return np.around(tn_mu, 5), p_value


if __name__ == '__main__':
    _, p_value = run()

    if p_value is not None:
        print('p-value:', p_value)
