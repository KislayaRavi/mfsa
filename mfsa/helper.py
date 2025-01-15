import numpy as np
from abc import abstractmethod

def get_initial_stats(f_list, num_samples, dim):
    samples = np.random.uniform(0, 1, (num_samples, dim))
    list_Y = []
    mean_array = np.zeros(len(f_list))
    var_array = np.zeros(len(f_list))
    var_diff_array = np.zeros(len(f_list))
    pearson_array = np.ones(len(f_list))
    g_term_array = np.zeros((len(f_list), num_samples))
    q_array, tau_array, delta_array = np.ones(len(f_list)), np.ones(len(f_list)), np.ones(len(f_list))
    for i, f in enumerate(f_list):
        Y = f(samples)
        mean_array[i] = np.mean(Y)
        var_array[i] = np.var(Y, ddof=1)
        delta_array[i] = np.mean((Y-mean_array[i])**4)
        g_term_array[i, :] = (Y - mean_array[i])**2
        tau_array[i] = np.std(g_term_array[i, :], ddof=1)
        list_Y.append(Y)
        if i==0:
            var_diff_array[i] = var_array[i]
        else:
            var_diff_array[i] = np.var(list_Y[i] - list_Y[i-1], ddof=1)
    for i in range(len(f_list)-1):
        pearson_array[i] = np.mean((list_Y[i] - mean_array[i])*(list_Y[-1] - mean_array[-1]))/(np.sqrt(var_array[i])*np.sqrt(var_array[-1]))
        q_array[i] = np.sum((g_term_array[i, :] - np.mean(g_term_array[i, :])) *(g_term_array[-1, :] - np.mean(g_term_array[-1, :])))/(num_samples-1)
        q_array[i] = q_array[i]/(tau_array[i]*tau_array[-1])
    return mean_array, var_array, pearson_array, var_diff_array, q_array, tau_array, delta_array

def get_unbiased_fourth_moment(biased_fourth_moment, var, N):
    t1 = N**2 - 3*N + 3
    t2 = (6*N - 9)*(N**2 - N)/t1
    t3 = (N**3)/N-1
    fm = (1/(t1 - (t2/N))) * ((t3*biased_fourth_moment) - (t2*var**2))
    return fm

def double_product_estimator(Y1, Y2):
    # Lemma 12 (Section 12.2) from Friedrich's thesis
    N = len(Y1)
    assert N == len(Y2), "Length of Y1 and Y2 should be same"
    biased_estimator = np.mean(Y1)*np.mean(Y2)
    unbiased_estimator = N*biased_estimator/(N-1) - np.mean(Y1*Y2)/(N-1)
    return unbiased_estimator

def triple_product_estimator(Y1, Y2, Y3):
    # Lemma 13 (Section 12.2) from Friedrich's thesis
    N = len(Y1)
    assert N == len(Y2) == len(Y3), "Length of Y1, Y2 and Y3 should be same"
    biased_estimator = np.mean(Y1)*np.mean(Y2)*np.mean(Y3)
    unbiased_estimator = (N**2 *biased_estimator)/((N-1)*(N-2)) - ((double_product_estimator(Y1*Y2, Y3) + double_product_estimator(Y1*Y3, Y2) + double_product_estimator(Y2*Y3, Y1))/(N-2))- (np.mean(Y1*Y2*Y3)/((N-1)*(N-2)))
    return unbiased_estimator

def quadruple_product_estimator(Y1, Y2):
    # product of the square of the means of both the random variables
    # Lemma 14 (Section 12.2) from Friedrich's thesis
    N = len(Y1)
    assert N == len(Y2), "Length of Y1, Y2 should be same"
    biased_estimator = np.mean(Y1)*np.mean(Y1)*np.mean(Y2)*np.mean(Y2)
    unbiased_estimator = (N**3 *biased_estimator)/((N-1)*(N-2)*(N-3)) - ((triple_product_estimator(Y1**2, Y2, Y2) + triple_product_estimator(Y1*Y2, Y1, Y2) + triple_product_estimator(Y1, Y1, Y2**2))/(N-3))- ((double_product_estimator(Y1**2, Y2**2) + 2*double_product_estimator(Y1*Y2, Y1*Y2) + 2*double_product_estimator(Y1**2 * Y2, Y2) + 2*double_product_estimator(Y1, Y2**2 * Y1))/((N-2)*(N-3))) - (np.mean(Y1**2 * Y2**2)/((N-1)*(N-2)*(N-3)))
    return unbiased_estimator

def variance_of_estimators_mfts(lf, hf, n_list, dim, num_samples=10000):
    Nlf, Nhf = n_list[0], n_list[1]
    samples = np.random.uniform(0, 1, (num_samples, dim))
    Ylf, Yhf = lf(samples), hf(samples)
    mean_lf, mean_hf = np.mean(Ylf), np.mean(Yhf)
    biased_fourth_moment_lf, biased_fourth_moment_hf = np.mean((Ylf - mean_lf)**4), np.mean((Yhf - mean_hf)**4)
    fourth_moment_lf, fourth_moment_hf = get_unbiased_fourth_moment(biased_fourth_moment_lf, np.var(Ylf, ddof=1), Nlf), get_unbiased_fourth_moment(biased_fourth_moment_hf, np.var(Yhf, ddof=1), Nhf)
    var_lf, var_hf = np.var(Ylf, ddof=1), np.var(Yhf, ddof=1)
    # For variance of variance estimator
    var_est_lf = (((Nlf - 1) * fourth_moment_lf) - ((Nlf - 3) * var_lf**2)) / (Nlf**2 - 2*Nlf + 3)
    var_est_hf = (((Nhf - 1) * fourth_moment_hf) - ((Nhf - 3) * var_hf**2)) / (Nhf**2 - 2*Nhf + 3)
    # var_est_lf = (1/Nlf)*(fourth_moment_lf - (Nlf-3)*var_lf**2/(Nlf-1))
    # var_est_hf = (1/Nhf)*(fourth_moment_hf - (Nhf-3)*var_hf**2/(Nhf-1))
    t1 = np.mean(Ylf**2 * Yhf**2)
    t2 = -2*double_product_estimator(Yhf**2 * Ylf, Ylf)
    t3 = 2*triple_product_estimator(Ylf, Ylf,Yhf**2)
    t4 = -2*double_product_estimator(Yhf, Ylf**2 * Yhf)
    t5 = 4*triple_product_estimator(Ylf, Yhf, Ylf*Yhf)
    t6 = 2*triple_product_estimator(Yhf, Yhf, Ylf**2)
    t7 = -4*quadruple_product_estimator(Ylf, Yhf)
    t8 = -1*double_product_estimator(Ylf**2, Yhf**2)
    cov1 = (t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8)/Nhf 
    cov2 = (double_product_estimator(Ylf*Yhf, Ylf*Yhf) - (2*triple_product_estimator(Ylf*Yhf, Ylf, Yhf)) + quadruple_product_estimator(Ylf, Yhf))/(Nhf*(Nhf-1))
    # print('Second order term', np.mean(Yhf*Ylf) - (2*mean_hf*mean_lf*np.mean(Yhf*Ylf)) - (mean_hf*mean_lf)**2)
    cov = cov1 + cov2
    total_var = var_est_lf + var_est_hf - 2*cov
    # print('Contribution of terms', var_est_lf, var_est_hf, 2*cov1, 2*cov2, total_var)
    # For variance of mean estimator
    # var_diff = np.var(Yhf - Ylf, ddof=1)
    # var_mean_estimator = (var_lf/Nlf) + (var_diff/Nhf)
    cov_lf_hf = np.sum((Ylf - mean_lf)*(Yhf - mean_hf))/(num_samples-1)
    var_mean_estimator = (var_lf / Nlf) + (var_lf / Nhf) + (var_hf / Nhf) - (2 * cov_lf_hf / Nhf)
    # print('Variance terms', (var_lf / Nlf) + (var_lf / Nhf) + (var_hf / Nhf) - (2 * cov_lf_hf / Nhf) - (var_hf*5/(6*Nhf+Nlf)), Nhf+Nlf)
    if total_var < 0:
        total_var = np.abs(total_var)
    return var_mean_estimator, total_var

def variance_of_estimators_sf(f, N, dim, num_samples=10000):
    samples = np.random.uniform(0, 1, (num_samples, dim))
    Y = f(samples)
    mean_f = np.mean(Y)
    # For variance of variance estimator
    fourth_moment_biased = np.mean((Y - mean_f)**4)
    var_f = np.var(Y, ddof=1)
    fourth_moment_f = get_unbiased_fourth_moment(fourth_moment_biased, var_f, num_samples)
    var_est_f = (((N - 1) * fourth_moment_f) - ((N - 3) * var_f**2)) / (N**2 - 2*N + 3)
    # var_est_f = (1/N)*(fourth_moment_f - (N-3)*var_f**2/(N-1))
    # For variance of mean estimator
    var_mean_estimator = var_f / N
    return var_mean_estimator, var_est_f
    
def variance_of_estimators_mfcv(lf, hf, n_array, dim, num_samples=10000):
    # alpha optimized is directly substituted here
    f_list = [lf, hf]
    num_fidelity = len(f_list)
    mean_array, var_array, pearson, var_diff_array, q_array, tau_array, delta_array = get_initial_stats(f_list, num_samples, dim)
    # For variance of mean estimator
    var_mean_estimator = (var_array[-1] / n_array[-1]) - np.sum([(1/n_array[i+1] - 1/n_array[i])*pearson[i] for i in range(len(f_list)-1)])*var_array[-1]
    # For variance of variance estimator
    t1 = q_array * tau_array * tau_array[-1]
    t2 = (pearson**2) * var_array * var_array[-1]
    var_var_estimator = (delta_array[-1] - (n_array[-1]-3)*(var_array[-1]**2)/(n_array[-1]-1))/n_array[-1]
    for i in range(num_fidelity-1):
        numerator = ((t1[i] + (2*t2[i]/(n_array[i]-1)))/n_array[i]) - ((t1[i] + (2*t2[i]/(n_array[i+1]-1)))/n_array[i+1])
        denominator = ((delta_array[i] - (n_array[i+1]-3)*var_array[i]**2/(n_array[i+1]-1))/n_array[i+1]) - ((delta_array[i] - (n_array[i]-3)*var_array[i]**2/(n_array[i]-1))/n_array[i])
        var_var_estimator -= numerator**2/denominator
    return var_mean_estimator, var_var_estimator


# def covariance_between_variance_estimators(hf, lf, num_samples, dim):
#     samples = np.random.uniform(0, 1, (num_samples, dim))
#     Y_hf = hf(samples)
#     Y_lf = lf(samples)
#     mean_Y_hf = np.mean(Y_hf)
#     mean_Y_lf = np.mean(Y_lf)
#     Y_lf_Y_hf = 

def get_list_samples_rank_stats(samples, Y, N, dim):
    assert samples.shape[0] == N
    assert samples.shape[1] == dim
    assert len(Y) == N
    list_Y = [Y]
    px = samples.argsort(axis=0)
    pi_j = px.argsort(axis=0) + 1
    argpiinv = (pi_j % N) + 1
    for i in range(dim):
        N_j = px[argpiinv[:, i] - 1, i]
        YN_j = Y[N_j]
        list_Y.append(YN_j)
    return list_Y

class IshigamiBase():

    def __init__(self, a, b, lower=-np.pi, upper=np.pi):
        self.lower, self.upper, self.dim = lower, upper, 3
        self.a, self.b = a, b
        self.num_eval_lf, self.num_eval_hf = 0, 0

    def transform_coordinates(self, x):
        return x * (self.upper - self.lower) + self.lower

    def hf(self, x): # this is the main Ishigami function
        temp = np.atleast_2d(x)
        if temp.shape[1] != self.dim:
            temp = temp.T
        q = self.transform_coordinates(temp)
        f = np.sin(q[:, 0]) + self.a * (np.sin(q[:, 1])**2) + (self.b * np.sin(q[:, 0]) * (q[:, 2]**4))
        self.num_eval_hf += len(temp)
        return f

    @abstractmethod
    def lf(self, x):
    	# This is just a dummy function to test multi-fidelity Ishigami toy problems :D
        temp = np.atleast_2d(x)
        if temp.shape[1] != self.dim:
            temp = temp.T
        q = self.transform_coordinates(temp)
        a, b = self.a +0.2, self.b+0.1
        f =( np.sin(q[:, 0]) + a * (np.sin(q[:, 1])**2) + (b * np.sin(q[:, 0]) * (q[:, 2]**4)) ) + 0.5
        self.num_eval_lf += len(temp)
        return f

    def calculate_statistics(self):
        mean = self.a * 0.5
        D1 = (self.b * np.pi**4 / 5) + (self.b**2 * np.pi**8 / 50) + 0.5
        D2 = self.a**2 / 8
        D13 = 19 * self.b**2 * np.pi**8 / 450
        D = D1 + D2 + D13
        assert np.abs(D - (D1 + D2 + D13)) < 0.001
        local_sobol = [D1/D, D2/D, 0.]
        global_sobol = [(D1+D13)/D, D2/D, D13/D]
        return mean, D, local_sobol, global_sobol
    
class Ishigami1(IshigamiBase):

    def lf(self, x):
        temp = np.atleast_2d(x)
        if temp.shape[1] != self.dim:
            temp = temp.T
        q = self.transform_coordinates(temp)
        a, b = self.a +0.1, self.b
        f =( np.sin(q[:, 0]) + a * (np.sin(q[:, 1])**2) + (b * np.sin(q[:, 0]) * (q[:, 2]**4)) ) + 0.02
        self.num_eval_lf += len(temp)
        return f