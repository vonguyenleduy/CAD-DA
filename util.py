import numpy as np
from scipy.optimize import linprog
from mpmath import mp

mp.dps = 500


def LP_solver(c_vec, S, u, G, h):
    # res = linprog(c_vec, A_ub=G, b_ub=h, A_eq=S, b_eq=u, method='simplex')
    res = linprog(c_vec, A_ub=G, b_ub=h, A_eq=S, b_eq=u, method='simplex',
                  options={'maxiter': 10000})
    return res


def construct_S_u_G_h(n, m):
    dim_t = n * m

    M_r = np.zeros((n, dim_t))

    for i in range(n):
        M_r[i, i * m:i * m + m] = np.ones(m)

    M_c = np.zeros((m, dim_t))

    for i in range(m):
        for j in range(i, dim_t, m):
            M_c[i, j] = 1.0

    S = np.vstack((M_r, M_c))
    u = np.vstack((np.ones((n, 1))/n, np.ones((m, 1))/m))

    # Remove any redundant row (e.g., last row)
    S = S[:-1, :]
    u = u[:-1, :]

    # Construct G
    G = -np.identity(dim_t)

    # Construct h
    h = np.zeros((dim_t, 1))

    return S, u, G, h


def construct_set_basis_non_basis_idx(t_hat):
    A = []
    Ac = []

    for i in range(len(t_hat)):
        if t_hat[i] != 0.0:
            A.append(i)
        else:
            Ac.append(i)

    return A, Ac


def construct_Omega(n, m):
    idx_matrix = np.identity(n)
    Omega = None
    for i in range(n):
        temp_vec = None
        for j in range(n):
            if idx_matrix[i][j] == 1.0:
                if j == 0:
                    temp_vec = np.ones((m, 1))
                else:
                    temp_vec = np.hstack((temp_vec, np.ones((m, 1))))
            else:
                if j == 0:
                    temp_vec = np.zeros((m, 1))
                else:
                    temp_vec = np.hstack((temp_vec, np.zeros((m, 1))))

        temp_vec = np.hstack((temp_vec, -np.identity(m)))

        if i == 0:
            Omega = temp_vec.copy()
        else:
            Omega = np.vstack((Omega, temp_vec))

    return Omega


def construct_list_kronecker_product(Omega, d):
    list_kronecker_product = []

    for k in range(d):
        e_d_k = np.zeros((d, 1))
        e_d_k[k][0] = 1.0

        kronecker_product = np.kron(Omega, e_d_k.T)
        list_kronecker_product.append(kronecker_product)

    return list_kronecker_product


def construct_Theta(n, m, d, data_obs):
    idx_matrix = np.identity(n)
    Omega = None
    for i in range(n):
        temp_vec = None
        for j in range(n):
            if idx_matrix[i][j] == 1.0:
                if j == 0:
                    temp_vec = np.ones((m, 1))
                else:
                    temp_vec = np.hstack((temp_vec, np.ones((m, 1))))
            else:
                if j == 0:
                    temp_vec = np.zeros((m, 1))
                else:
                    temp_vec = np.hstack((temp_vec, np.zeros((m, 1))))

        temp_vec = np.hstack((temp_vec, -np.identity(m)))

        if i == 0:
            Omega = temp_vec.copy()
        else:
            Omega = np.vstack((Omega, temp_vec))

    Theta = np.zeros((n * m, n * d + m * d))

    list_sign = []
    list_kronecker_product = []

    for k in range(d):
        e_d_k = np.zeros((d, 1))
        e_d_k[k][0] = 1.0

        kronecker_product = np.kron(Omega, e_d_k.T)
        dot_product = np.dot(kronecker_product, data_obs)
        s_k = np.sign(dot_product)

        Theta = Theta + s_k * kronecker_product

        list_sign.append(s_k)
        list_kronecker_product.append(kronecker_product)

    return Theta, list_sign, list_kronecker_product


def intersect(range_1, range_2):
    lower = max(range_1[0], range_2[0])
    upper = min(range_1[1], range_2[1])

    if upper < lower:
        return []
    else:
        return [lower, upper]


def intersect_interval(initial_range, list_2_range):
    if len(initial_range) == 0:
        return []

    final_list = [initial_range]

    for each_2_range in list_2_range:

        lower_range = [np.NINF, each_2_range[0]]
        upper_range = [each_2_range[1], np.Inf]

        new_final_list = []

        for each_1_range in final_list:
            local_range_1 = intersect(each_1_range, lower_range)
            local_range_2 = intersect(each_1_range, upper_range)

            if len(local_range_1) > 0:
                new_final_list.append(local_range_1)

            if len(local_range_2) > 0:
                new_final_list.append(local_range_2)

        final_list = new_final_list

    return final_list


def find_interval_lp(list_se_u_0, list_se_u_1, list_se_u_2):
    list_se_u_0 = np.around(list_se_u_0, 10)
    list_se_u_1 = np.around(list_se_u_1, 10)
    list_se_u_2 = np.around(list_se_u_2, 10)

    L_prime = np.NINF
    U_prime = np.Inf

    L_tilde = np.NINF
    U_tilde = np.Inf

    list_2_interval = []

    for i in range(len(list_se_u_0)):
        c = - list_se_u_0[i]
        b = - list_se_u_1[i]
        a = - list_se_u_2[i]

        if a == 0:
            if b == 0:
                if c > 0:
                    print('Error a = 0, b = 0, c > 0')

            elif b < 0:
                temporal_lower_bound = - c / b
                L_prime = max(L_prime, temporal_lower_bound)

            elif b > 0:
                temporal_upper_bound = - c / b
                U_prime = min(U_prime, temporal_upper_bound)
        else:
            delta = b ** 2 - 4 * a * c

            delta = np.around(delta, 10)

            if delta == 0:
                if a > 0:
                    print('c_T_A_c > 0 and delta = 0')
            elif delta < 0:
                if a > 0:
                    print('c_T_A_c > 0 and delta < 0')
            elif delta > 0:
                if a > 0:
                    x_lower = (-b - np.sqrt(delta)) / (2 * a)
                    x_upper = (-b + np.sqrt(delta)) / (2 * a)

                    if x_lower > x_upper:
                        print('x_lower > x_upper')

                    L_tilde = max(L_tilde, x_lower)
                    U_tilde = min(U_tilde, x_upper)

                else:
                    x_1 = (-b - np.sqrt(delta)) / (2 * a)
                    x_2 = (-b + np.sqrt(delta)) / (2 * a)

                    x_low = min(x_1, x_2)
                    x_up = max(x_1, x_2)
                    list_2_interval.append([x_low, x_up])

    # final_list_interval = intersect_interval(
    #     intersect([L_prime, U_prime], [L_tilde, U_tilde]),
    #     list_2_interval)

    # return final_list_interval

    list_1_interval = [intersect([L_prime, U_prime], [L_tilde, U_tilde])]

    return list_1_interval, list_2_interval


def compute_a_b(data, eta, dim_data):
    sq_norm = (np.linalg.norm(eta))**2

    e1 = np.identity(dim_data) - (np.dot(eta, eta.T))/sq_norm
    a = np.dot(e1, data)

    b = eta/sq_norm

    return a, b


def pivot_with_constructed_interval(z_interval, eta, etaTy, cov, tn_mu):
    tn_sigma = np.sqrt(np.dot(np.dot(eta.T, cov), eta))[0][0]
    # print(tn_sigma)
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etaTy >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etaTy >= al) and (etaTy < ar):
            numerator = numerator + mp.ncdf((etaTy - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None