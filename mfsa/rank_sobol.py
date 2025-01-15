import numpy as np
from mfsa.helper import get_initial_stats
from mfsa.helper import *

def set_random_seed():
    seed = np.random.randint(0, 1000)
    np.random.seed(seed)

def get_cross_correlation_array(list_Y, mean, dim):
    assert len(list_Y) == dim+1
    cross_correlation_array = np.zeros(dim)
    for i in range(dim):
        cross_correlation_array[i] = np.mean(list_Y[0]*list_Y[i+1]) - mean**2
        # cross_correlation_array[i] = np.sum((list_Y[0] - mean)*(list_Y[i+1] - np.mean(list_Y[i+1])))/(len(list_Y[0])-1)
    # cross_correlation_array[cross_correlation_array < 0] = 0
    return cross_correlation_array

def sf_rank(func, num_samples, dim, samples=None):
    if samples is None:
        samples = np.random.uniform(0, 1, (num_samples, dim)) # code desingned to work with samples in [0, 1]
    Y = func(samples)
    list_Y = get_list_samples_rank_stats(samples, Y, num_samples, dim)
    mean = np.mean(list_Y[0])
    var = np.var(list_Y[0], ddof=1)
    # var = np.mean(list_Y[0]**2) - mean**2
    first_order_sobol = get_cross_correlation_array(list_Y, mean, dim)/var
    first_order_sobol[first_order_sobol < 0] = 0 # this is a superfluos step, comment it out if you want
    first_order_sobol[first_order_sobol > 1] = 1
    return mean, var, first_order_sobol

def mfts_telescopic_terms(lf, hf, samples, num_samples, dim):
    Ylf, Yhf = lf(samples), hf(samples)
    list_Ylf = get_list_samples_rank_stats(samples, Ylf, num_samples, dim)
    list_Yhf = get_list_samples_rank_stats(samples, Yhf, num_samples, dim)
    mean_lf, mean_hf = np.mean(Ylf), np.mean(Yhf)
    mean = mean_hf - mean_lf
    var_lf, var_hf = np.var(Ylf, ddof=1), np.var(Yhf, ddof=1)
    var = np.sum((Yhf - mean_hf)**2 - (Ylf - mean_lf)**2) / (num_samples - 1)
    sobol_hf = get_cross_correlation_array(list_Yhf, mean_hf, dim)/var_hf
    sobol_lf = get_cross_correlation_array(list_Ylf, mean_lf, dim)/var_lf
    sobol = sobol_hf - sobol_lf
    sobol[sobol < 0] = 0
    sobol[sobol > 1] = 1
    return mean, var, sobol

def mf_sample_size_variance_constraint(var_array, cost_array, error_tol):
    assert len(var_array) == len(cost_array)
    num_f = len(var_array)
    ct = np.zeros(num_f)
    ct[0] = cost_array[0]
    for i in range(1,num_f):
        ct[i] = cost_array[i] + cost_array[i-1]
    constant_term = np.sum(var_array*ct)/error_tol
    sample_size = constant_term * np.sqrt(var_array/ct)
    sample_size = np.array(sample_size, dtype=int)
    sample_size[sample_size ==1] = 2
    return sample_size

def mf_sample_size_budget_constraint(var_array, cost_array, computational_budget):
    assert len(var_array) == len(cost_array)
    num_f = len(var_array)
    ct = np.zeros(num_f)
    ct[0] = cost_array[0]
    for i in range(1,num_f):
        ct[i] = cost_array[i] + cost_array[i-1]
    # constant_term = np.sum(np.sqrt(var_array*ct))**2/computational_budget**2
    # sample_size = np.sqrt(var_array/(ct*constant_term))
    constant_term = computational_budget/np.sum(np.sqrt(var_array*ct))
    sample_size = constant_term * np.sqrt(var_array/ct)
    sample_size = np.array(sample_size, dtype=int)
    sample_size[sample_size ==1] = 2
    return sample_size

def mf_rank_biased(f_list, n_list, dim):
    assert len(f_list) > 0
    assert len(f_list) == len(n_list)
    num_f = len(f_list)
    mean_array, var_array = np.zeros(num_f), np.zeros(num_f)
    cross_correlation_matrix = np.zeros((num_f, dim))
    for i, n in enumerate(n_list):
        assert n > 0
        samples = np.random.uniform(0, 1, (n, dim))
        if i == 0:
            Y = f_list[i](samples)
            list_Y = get_list_samples_rank_stats(samples, Y, n, dim)
            mean_array[i] = np.mean(list_Y[0])
            var_array[i] = np.var(list_Y[0], ddof=1)
            cross_correlation_matrix[i] = get_cross_correlation_array(list_Y, mean_array[i], dim)
        else:
            Y_i = f_list[i](samples)
            Y_i1 = f_list[i-1](samples)
            list_Y_i = get_list_samples_rank_stats(samples, Y_i, n, dim)
            list_Y_i1 = get_list_samples_rank_stats(samples, Y_i1, n, dim)
            mean_hf, mean_lf = np.mean(list_Y_i[0]), np.mean(list_Y_i1[0])
            var_hf, var_lf = np.var(list_Y_i[0], ddof=1), np.var(list_Y_i1[0], ddof=1)
            cross_correlation_hf = get_cross_correlation_array(list_Y_i, mean_hf, dim)
            cross_correlation_lf = get_cross_correlation_array(list_Y_i1, mean_lf, dim)
            # Y_k_i, Y_k_i1 = np.array(list_Y_i[1:]), np.array(list_Y_i1[1:])
            # t1 = (np.matmul(Y_k_i, Y_i) - np.matmul(Y_k_i1, Y_i1))/n
            # t2 = np.mean(Y_i**2 - Y_i1**2)
            # cross_correlation_matrix[i] = t1 - t2
            mean_array[i] = mean_hf - mean_lf
            var_array[i] = var_hf - var_lf
            cross_correlation_matrix[i] = cross_correlation_hf - cross_correlation_lf
            # cross_correlation_matrix[i] = get_cross_correlation_array(np.array(list_Y_i) - np.array(list_Y_i1), mean_hf-mean_lf,dim)
    mean = np.sum(mean_array)
    var = np.sum(var_array)
    first_order_sobol = np.sum(cross_correlation_matrix, axis=0)/var
    first_order_sobol[first_order_sobol < 0] = 0
    first_order_sobol[first_order_sobol > 1] = 1
    return mean, var, first_order_sobol

def mf_rank_unbiased(f_list, n_list, dim):
    assert len(f_list) > 0
    assert len(f_list) == len(n_list)
    num_f = len(f_list)
    mean_array, var_array = np.zeros(num_f), np.zeros(num_f)
    first_order_sobol_levels = np.zeros((num_f, dim))
    for i, n in enumerate(n_list):
        assert n > 0
        if i == 0:
            mean_array[i], var_array[i], first_order_sobol_levels[i] = sf_rank(f_list[i], n, dim)
        else:
            samples = np.random.uniform(0, 1, (n, dim))
            mean_array[i], var_array[i], first_order_sobol_levels[i, :] = mfts_telescopic_terms(f_list[i-1], f_list[i], samples, n, dim)
            # m1, v1, s1 = sf_rank(f_list[i], n, dim, samples=samples)
            # m2, v2, s2 = sf_rank(f_list[i-1], n, dim, samples=samples)
            # mean_array[i] = m1 - m2
            # var_array[i] = v1 - v2
            # first_order_sobol_levels[i, :] = s1 - s2
    mean = np.sum(mean_array)
    var = np.sum(var_array)
    first_order_sobol = np.sum(first_order_sobol_levels, axis=0)
    first_order_sobol[first_order_sobol < 0] = 0
    first_order_sobol[first_order_sobol > 1] = 1
    return mean, var, first_order_sobol

def mf_cv_sample_size(var_array, pearson_array, cost_array, total_budget):
    num_f = len(var_array)
    pa = np.zeros(num_f+1)
    pa[1:] = pearson_array
    r = np.zeros(num_f)
    for i in range(num_f):
        r[i] = np.sqrt((cost_array[-1]*(pa[i+1]**2 - pa[i]**2))/(cost_array[i]*(1-pearson_array[-2]**2)))
    sample_size = np.zeros(num_f)
    sample_size[-1] = total_budget/np.sum(cost_array*r)
    for i in range(2, num_f+1):
        sample_size[-i] = r[-i]*sample_size[-i+1]
    alpha = pearson_array * np.sqrt(var_array[-1])/np.sqrt(var_array)
    sample_size = np.array(sample_size, dtype=int)
    sample_size[sample_size ==1] = 2
    return sample_size, alpha

def get_alpha_cv_var_red(n_array, var_array, pearson, q_array, tau_array, delta_array):
    num_fidelity = len(n_array)
    alpha = np.ones(num_fidelity)
    t1 = q_array * tau_array * tau_array[-1]
    t2 = (pearson**2) * var_array * var_array[-1]
    for i in range(num_fidelity-1):
        numerator = ((t1[i] + (2*t2[i]/(n_array[i+1]-1)))/n_array[i+1]) - ((t1[i] + (2*t2[i]/(n_array[i]-1)))/n_array[i])
        denominator = ((delta_array[i] - (n_array[i+1]-3)*var_array[i]**2/(n_array[i+1]-1))/n_array[i+1]) - ((delta_array[i] - (n_array[i]-3)*var_array[i]**2/(n_array[i]-1))/n_array[i])
        alpha[i] = numerator / denominator
    return alpha


def mf_rank_cv(f_list, n_list, dim, alpha_array, alpha_var_estimator):
    assert len(f_list) > 0
    assert len(f_list) == len(n_list)
    num_f = len(f_list)
    mean_array, var_array = np.zeros(num_f), np.zeros(num_f)
    # alpha_array = np.ones(num_f)
    sobol_index_matrix = np.zeros((num_f, dim))
    all_samples = np.random.uniform(0,1,(n_list[0],dim))
    for i, n in enumerate(n_list):
        samples = all_samples[:n, :]
        Yi = f_list[i](samples)
        list_Yi = get_list_samples_rank_stats(samples, Yi, n, dim)
        mean_array[i] = np.mean(list_Yi[0])
        var_array[i] = np.var(list_Yi[0], ddof=1)
        sobol_index_matrix[i] = get_cross_correlation_array(list_Yi, mean_array[i], dim)/var_array[i]
        if i > 0:
            Yi1 = f_list[i-1](samples)
            mean_Yi1 = np.mean(Yi1)
            var_Yi1 = np.var(Yi1, ddof=1)
            list_Yi1 = get_list_samples_rank_stats(samples, Yi1, n, dim)
            temp_sobol = get_cross_correlation_array(list_Yi1, mean_Yi1, dim)/var_Yi1#var_array[i-1]
            # alpha_array[i-1] = np.mean((Yi - mean_array[i])*(Yi1 - mean_Yi1))*var_array[i]/var_array[i-1]
            mean_array[i-1] -= mean_Yi1
            var_array[i-1] -= var_Yi1
            sobol_index_matrix[i-1] -= temp_sobol   
    mean = np.sum(mean_array*alpha_array)
    var = np.sum(var_array*(alpha_var_estimator))
    first_order_sobol = np.zeros(dim)
    for i in range(num_f):
        first_order_sobol += alpha_var_estimator[i]*sobol_index_matrix[i, :]
    # first_order_sobol = np.matmul(alpha_array, sobol_index_matrix)
    first_order_sobol[first_order_sobol < 0] = 0
    first_order_sobol[first_order_sobol > 1] = 1
    return mean, var, first_order_sobol

if __name__ == '__main__':
    ishigami = Ishigami1(1, 0.01)
    dim, cost_array = 3, np.array([1, 5])
    mean_actual, var_actual, local_sobol_actual, global_sobol_actual = ishigami.calculate_statistics()
    f_list = [ishigami.lf, ishigami.hf]
    print('Actual stats:', mean_actual, var_actual, local_sobol_actual)
    # Sample size tester
    _, var_array, pearson_array, var_diff_array, q_array, tau_array, delta_array = get_initial_stats(f_list, 100, dim)
    mf_num_samples = mf_sample_size_variance_constraint(var_diff_array, cost_array, 0.01)
    print('MF sample size:', mf_num_samples)
    computational_budget = int(np.sum(mf_num_samples*cost_array))
    print('Computational budget:', computational_budget)
    # CV- sample size tester
    mf_num_samples_cv, alpha = mf_cv_sample_size(var_array, pearson_array, cost_array, computational_budget)
    print('MF-CV sample size:', mf_num_samples_cv)
    print('Alpha mean estimator:', alpha)
    alpha_var_estimator = get_alpha_cv_var_red(mf_num_samples_cv, var_array, pearson_array, q_array, tau_array, delta_array)
    print('Alpha var estimator:', alpha_var_estimator)
    # SF
    N = computational_budget
    mean, var, first_order_sobol = sf_rank(ishigami.hf, N, dim)
    print('SF stats:', mean, var, first_order_sobol)
    mean_lf, var_lf, first_order_sobol_lf = sf_rank(ishigami.lf, N, dim)
    print('SF stats for low fidelity:', mean_lf, var_lf, first_order_sobol_lf)
    # MF biased sobol indices
    n_list = mf_num_samples
    mean, var, first_order_sobol = mf_rank_biased(f_list, n_list, dim)
    print('MF stats with biased sobol indices:', mean, var, first_order_sobol)
    # MF unbiased sobol indices
    n_list = mf_num_samples
    mean, var, first_order_sobol = mf_rank_unbiased(f_list, n_list, dim)
    print('MF stats with unbiased sobol indices:', mean, var, first_order_sobol)
    # MF-CV
    mean_cv, var_cv, first_order_sobol_cv = mf_rank_cv(f_list, mf_num_samples_cv, dim, alpha, alpha_var_estimator)
    print('MF-CV stats:', mean_cv, var_cv, first_order_sobol_cv)
    

    
