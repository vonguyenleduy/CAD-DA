import numpy as np
import util


def construct_e_vector(index, n_s, n_t, T_hat):
    if index < n_s:
        e_i = n_s * T_hat[index, :]
        e_i = e_i.reshape((n_t, 1))
        e_i = np.vstack((np.zeros((n_s, 1)), e_i))
    else:
        e_i = np.zeros((n_s + n_t, 1))
        e_i[index] = 1.0

    return e_i


def over_conditioning(Xs_obs, Xt_obs, n_s, n_t, d, alpha, a, b):
    dim_t = n_s * n_t

    # Cost matrix
    C = np.zeros((n_s, n_t))

    for i in range(n_s):
        e_x = Xs_obs[i, :]
        for j in range(n_t):
            e_y = Xt_obs[j, :]
            C[i, j] = np.sum((e_x - e_y) ** 2)

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

    list_u = []
    list_v = []

    for each_dim in range(d):
        merged_data = merge_matrix[:, each_dim]

        median_1 = np.median(merged_data)
        list_abs_deviation = np.abs(merged_data - median_1)
        median_2 = np.median(list_abs_deviation)

        lower = median_1 - alpha * median_2
        upper = median_1 + alpha * median_2

        dim_vector = np.zeros((d, 1))
        dim_vector[each_dim] = 1.0

        idx_median_1 = np.where(merged_data == median_1)[0][0]
        idx_median_2 = np.where(list_abs_deviation == median_2)[0][0]

        e_idx_med_1 = construct_e_vector(idx_median_1, n_s, n_t, T_hat)

        e_idx_med_1 = np.kron(e_idx_med_1, dim_vector)

        for i in range(n_s + n_t):
            element = merged_data[i]

            e_i = construct_e_vector(i, n_s, n_t, T_hat)
            e_i = np.kron(e_i, dim_vector)

            if median_1 > element:
                u = np.dot(e_i.T, b)[0][0] - np.dot(e_idx_med_1.T, b)[0][0]
                v = np.dot(e_idx_med_1.T, a)[0][0] - np.dot(e_i.T, a)[0][0]
            else:
                u = np.dot(e_idx_med_1.T, b)[0][0] - np.dot(e_i.T, b)[0][0]
                v = np.dot(e_i.T, a)[0][0] - np.dot(e_idx_med_1.T, a)[0][0]

            list_u.append(u)
            list_v.append(v)

        e_idx_med_2 = construct_e_vector(idx_median_2, n_s, n_t, T_hat)
        e_idx_med_2 = np.kron(e_idx_med_2, dim_vector)

        sign_med1_med2 = np.sign(merged_data[idx_median_1] - merged_data[idx_median_2])

        h_1 = sign_med1_med2 * (e_idx_med_1 - e_idx_med_2)

        for i in range(n_s + n_t):
            sign_i = np.sign(median_1 - merged_data[i])

            e_i = construct_e_vector(i, n_s, n_t, T_hat)
            e_i = np.kron(e_i, dim_vector)

            h_2 = sign_i * (e_idx_med_1 - e_i)

            if median_2 > list_abs_deviation[i]:
                u = np.dot(h_2.T, b)[0][0] - np.dot(h_1.T, b)[0][0]
                v = np.dot(h_1.T, a)[0][0] - np.dot(h_2.T, a)[0][0]
            else:
                u = np.dot(h_1.T, b)[0][0] - np.dot(h_2.T, b)[0][0]
                v = np.dot(h_2.T, a)[0][0] - np.dot(h_1.T, a)[0][0]

            list_u.append(u)
            list_v.append(v)

        # Selection event for identifying outlier
        for j in range(n_t):

            new_idx_j = n_s + j
            e_j_selected = construct_e_vector(new_idx_j, n_s, n_t, T_hat)

            e_j_selected = np.kron(e_j_selected, dim_vector)

            vector_upper = e_idx_med_1 + alpha * sign_med1_med2 * (e_idx_med_1 - e_idx_med_2)
            u_upper = np.dot(vector_upper.T, b)[0][0] - np.dot(e_j_selected.T, b)[0][0]
            v_upper = np.dot(e_j_selected.T, a)[0][0] - np.dot(vector_upper.T, a)[0][0]

            if merged_data[new_idx_j] > upper:
                list_u.append(u_upper)
                list_v.append(v_upper)
            else:
                list_u.append(-u_upper)
                list_v.append(-v_upper)

            vector_lower = e_idx_med_1 - alpha * sign_med1_med2 * (e_idx_med_1 - e_idx_med_2)
            u_lower = np.dot(e_j_selected.T, b)[0][0] - np.dot(vector_lower.T, b)[0][0]
            v_lower = np.dot(vector_lower.T, a)[0][0] - np.dot(e_j_selected.T, a)[0][0]

            if merged_data[new_idx_j] < lower:
                list_u.append(u_lower)
                list_v.append(v_lower)
            else:
                list_u.append(-u_lower)
                list_v.append(-v_lower)

    Vminus = np.NINF
    Vplus = np.Inf

    # Selection event for Outlier and Sign
    for i in range(len(list_u)):
        a_coef = np.around(list_u[i], 5)
        b_coef = np.around(list_v[i], 5)

        if a_coef == 0.0:
            if b_coef < 0:
                continue
                # print('error')

            continue

        temp = b_coef / a_coef

        if a_coef > 0:
            Vplus = min(Vplus, temp)
        elif a_coef < 0:
            Vminus = max(Vminus, temp)

    if Vminus > Vplus:
        print("error")

    # Find interval LP
    Omega = util.construct_Omega(n_s, n_t)
    list_kronecker_product = util.construct_list_kronecker_product(Omega, d)

    u = np.zeros((dim_t, 1))
    v = np.zeros((dim_t, 1))
    w = np.zeros((dim_t, 1))

    for kronecker_product in list_kronecker_product:
        Omega_a = np.dot(kronecker_product, a)
        Omega_b = np.dot(kronecker_product, b)

        u = u + Omega_a ** 2
        v = v + 2 * Omega_a * Omega_b
        w = w + Omega_b ** 2

    u_A = u[A, :].copy()
    u_Ac = u[Ac, :].copy()

    v_A = v[A, :].copy()
    v_Ac = v[Ac, :].copy()

    w_A = w[A, :].copy()
    w_Ac = w[Ac, :].copy()

    S_ast_A = S[:, A].copy()
    S_ast_A_inv = np.linalg.inv(S_ast_A)
    S_ast_Ac = S[:, Ac].copy()

    S_ast_A_inv_S_ast_Ac = np.dot(S_ast_A_inv, S_ast_Ac)

    u_til = u_Ac.T - np.dot(u_A.T, S_ast_A_inv_S_ast_Ac)
    v_til = v_Ac.T - np.dot(v_A.T, S_ast_A_inv_S_ast_Ac)
    w_til = w_Ac.T - np.dot(w_A.T, S_ast_A_inv_S_ast_Ac)

    u_til = (np.around(u_til.flatten(), 10)).tolist()
    v_til = (np.around(v_til.flatten(), 10)).tolist()
    w_til = (np.around(w_til.flatten(), 10)).tolist()

    list_1_interval, list_2_interval = util.find_interval_lp(u_til, v_til, w_til)

    # if len(list_1_interval[0]) == 0:
    #     return [], detect_outlier_idx

    final_list_interval = util.intersect_interval(
        util.intersect(list_1_interval[0], [Vminus, Vplus]),
        list_2_interval)

    return final_list_interval[0][1], detect_outlier_idx


def run_parametric_si(a, b, n_s, n_t, d, alpha, threshold):
    zk = - threshold
    list_zk = [zk]
    list_outlier_idx = []

    # zk = 1.305614787630861
    # data_zk = a + b * zk
    # Xs_zk_vec = data_zk[0:n_s * d, :]
    # Xt_zk_vec = data_zk[n_s * d:, :]
    #
    # Xs_zk = Xs_zk_vec.reshape((n_s, d))
    # Xt_zk = Xt_zk_vec.reshape((n_t, d))
    # over_conditioning(Xs_zk, Xt_zk, n_s, n_t, d, alpha, a, b)

    while zk < threshold:
        data_zk = a + b * zk
        Xs_zk_vec = data_zk[0:n_s * d, :]
        Xt_zk_vec = data_zk[n_s * d:, :]

        Xs_zk = Xs_zk_vec.reshape((n_s, d))
        Xt_zk = Xt_zk_vec.reshape((n_t, d))

        next_zk, detect_outlier_idx_zk = over_conditioning(Xs_zk, Xt_zk, n_s, n_t, d, alpha, a, b)

        zk = next_zk + 0.0001

        # print(zk, detect_outlier_idx_zk)

        if zk < threshold:
            list_zk.append(zk)
        else:
            list_zk.append(threshold)

        list_outlier_idx.append(detect_outlier_idx_zk)

    return list_zk, list_outlier_idx


