import numpy as np
from mfsa.helper import *
from mfsa.rank_sobol import *
from mfsa.benchmarks import *
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
# import seaborn as sns
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# plt.style.use('ggplot')
SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 25
rc('font', size=MEDIUM_SIZE)          # controls default text sizes
rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure titles


def statistical_study(num_repetitions, type_plot='boxplot'):
    num_samples_init_study = 500
    dim, cost_array = 3, np.array([1, 5])
    benchmark = Ishigami(1, 0.01)
    # benchmark = Gfunction(dim, [10,5,3])
    # benchmark = C2(dim)
    # dim= 10
    # benchmark = Gstarfunction(dim, np.linspace(1, 10, dim))
    mean_actual, var_actual, local_sobol_actual, global_sobol_actual = benchmark.calculate_statistics()
    f_list = [benchmark.lf, benchmark.hf]
    print('Actual stats:', mean_actual, var_actual, local_sobol_actual)
    # Sample size tester
    np.random.seed(0)
    _, var_array, pearson_array, var_diff_array, q_array, tau_array, delta_array = get_initial_stats(f_list, num_samples_init_study, dim)
    # mf_num_samples = mf_sample_size_variance_constraint(var_array, cost_array, 0.1)
    mf_num_samples = mf_sample_size_budget_constraint(var_diff_array, cost_array, 500)
    print('MF sample size:', mf_num_samples)
    computational_budget = int(np.sum(mf_num_samples*cost_array))
    print('Computational budget:', computational_budget)
    # CV- sample size tester
    mf_num_samples_cv, alpha = mf_cv_sample_size(var_array, pearson_array, cost_array, computational_budget)
    print('MF-CV sample size:', mf_num_samples_cv)
    print('Alpha:', alpha)
    alpha_var_estimator = get_alpha_cv_var_red(mf_num_samples_cv, var_array, pearson_array, q_array, tau_array, delta_array)
    print('Alpha var estimator:', alpha_var_estimator)
    mean_sf_array, var_sf_array, first_order_sobol_sf_array = np.zeros(num_repetitions), np.zeros(num_repetitions), np.zeros((num_repetitions, dim))
    mean_mf_array_biased, var_mf_array_biased, first_order_sobol_mf_array_biased = np.zeros(num_repetitions), np.zeros(num_repetitions), np.zeros((num_repetitions, dim))
    mean_mf_array_unbiased, var_mf_array_unbiased, first_order_sobol_mf_array_unbiased = np.zeros(num_repetitions), np.zeros(num_repetitions), np.zeros((num_repetitions, dim))
    mean_mf_cv_array, var_mf_cv_array, first_order_sobol_mf_cv_array = np.zeros(num_repetitions), np.zeros(num_repetitions), np.zeros((num_repetitions, dim))
    for i in range(num_repetitions):
        np.random.seed(i)
        mean_sf_array[i], var_sf_array[i], first_order_sobol_sf_array[i,:] = sf_rank(benchmark.hf, int(computational_budget/cost_array[-1]), dim)
        np.random.seed(i)
        mean_mf_array_biased[i], var_mf_array_biased[i], first_order_sobol_mf_array_biased[i,:] = mf_rank_biased(f_list, mf_num_samples, dim)
        np.random.seed(i)
        mean_mf_array_unbiased[i], var_mf_array_unbiased[i], first_order_sobol_mf_array_unbiased[i,:] = mf_rank_unbiased(f_list, mf_num_samples, dim)
        np.random.seed(i)
        mean_mf_cv_array[i], var_mf_cv_array[i], first_order_sobol_mf_cv_array[i,:] = mf_rank_cv(f_list, mf_num_samples_cv, dim, alpha, alpha_var_estimator)
    error_mean_sf, error_var_sf, error_sobol_sf = mean_sf_array - mean_actual, var_sf_array - var_actual, first_order_sobol_sf_array - local_sobol_actual
    error_mean_mf_biased, error_var_mf_biased, error_sobol_mf_biased = mean_mf_array_biased - mean_actual, var_mf_array_biased - var_actual, first_order_sobol_mf_array_biased - local_sobol_actual
    error_mean_mf_unbiased, error_var_mf_unbiased, error_sobol_mf_unbiased = mean_mf_array_unbiased - mean_actual, var_mf_array_unbiased - var_actual, first_order_sobol_mf_array_unbiased - local_sobol_actual
    error_mean_mf_cv, error_var_mf_cv, error_sobol_mf_cv = mean_mf_cv_array - mean_actual, var_mf_cv_array - var_actual, first_order_sobol_mf_cv_array - local_sobol_actual
    print('Mean of error in mean estimator in order SF, MF biased, MF unbiased, MFCV', np.mean(np.abs(error_mean_sf)), np.mean(np.abs(error_mean_mf_biased)), np.mean(np.abs(error_mean_mf_unbiased)), np.mean(np.abs(error_mean_mf_cv)))
    print('Variance of mean estimator in order SF, MF biased, MF unbiased, MFCV', np.var(mean_sf_array, ddof=1), np.var(mean_mf_array_biased, ddof=1), np.var(mean_mf_array_unbiased, ddof=1), np.var(mean_mf_cv_array, ddof=1))
    print('Mean of error in variance estimator in order SF, MF biased, MF unbiased, MFCV', np.mean(np.abs(error_var_sf)), np.mean(np.abs(error_var_mf_biased)), np.mean(np.abs(error_var_mf_unbiased)), np.mean(np.abs(error_var_mf_cv)))
    print('Variance of variance estimator in order SF, MF biased, unbiased, MFCV', np.var(var_sf_array, ddof=1), np.var(var_mf_array_biased, ddof=1), np.var(var_mf_array_unbiased, ddof=1), np.var(var_mf_cv_array, ddof=1))
    print('Mean of error in Sobol estimator in order SF, MF biased, MF unbiased, MFCV', np.mean(np.abs(error_sobol_sf), axis=0), np.mean(np.abs(error_sobol_mf_biased), axis=0), np.mean(np.abs(error_sobol_mf_unbiased), axis=0), np.mean(np.abs(error_sobol_mf_cv), axis=0))
    print('Variance of Sobol estimator in order SF, MF biased, MF unbiased, MFCV', np.var(first_order_sobol_sf_array, axis=0, ddof=1), np.var(first_order_sobol_mf_array_biased, axis=0, ddof=1), np.var(first_order_sobol_mf_array_unbiased, axis=0, ddof=1), np.var(first_order_sobol_mf_cv_array, axis=0, ddof=1))
    # Plotting
    if type_plot == 'boxplot':
        fig, ax = plt.subplots(2, 3, figsize=(10, 15))
        ax[0,0].boxplot([error_mean_sf, error_mean_mf_biased, error_mean_mf_unbiased, error_mean_mf_cv], labels=['SF', 'MF biased', 'MF unbiased', 'MF-CV'], meanline=True)
        ax[0,0].axhline(0, color='black', linestyle='dotted')
        ax[0,0].set_title('Error in mean estimator')
        ax[0,1].boxplot([error_var_sf, error_var_mf_biased, error_var_mf_unbiased, error_var_mf_cv], labels=['SF', 'MF biased', 'MF unbiased', 'MF-CV'], meanline=True)
        ax[0,1].axhline(0, color='black', linestyle='dotted')
        ax[0,1].set_title('Error in variance estimator')
        ax[0,2].boxplot([error_sobol_sf[:, 0], error_sobol_mf_biased[:, 0], error_sobol_mf_unbiased[:, 0], error_sobol_mf_cv[:, 0]], labels=['SF', 'MF biased', 'MF unbiased', 'MF-CV'], meanline=True)
        ax[0,2].axhline(0, color='black', linestyle='dotted')
        ax[0,2].set_title('Error in Sobol estimator for $X_1$')
        ax[1,0].boxplot([error_sobol_sf[:, 1], error_sobol_mf_biased[:, 1], error_sobol_mf_unbiased[:, 1], error_sobol_mf_cv[:, 1]], labels=['SF', 'MF biased', 'MF unbiased', 'MF-CV'], meanline=True)
        ax[1,0].axhline(0, color='black', linestyle='dotted')
        ax[1,0].set_title('Error in Sobol estimator for $X_2$')
        ax[1,1].boxplot([error_sobol_sf[:, 2], error_sobol_mf_biased[:, 2], error_sobol_mf_unbiased[:, 2], error_sobol_mf_cv[:, 2]], labels=['SF', 'MF biased', 'MF unbiased', 'MF-CV'], meanline=True)
        ax[1,1].axhline(0, color='black', linestyle='dotted')
        ax[1,1].set_title('Error in Sobol estimator for $X_3$')
        plt.yscale('symlog')
        plt.show()

    # Same but with violin plots
    elif type_plot == 'violinplot': 
        fig, ax = plt.subplots(2, 3, figsize=(10, 15))
        ax[0,0].violinplot([error_mean_sf, error_mean_mf_biased, error_mean_mf_unbiased, error_mean_mf_cv])
        ax[0,0].axhline(0, color='black', linestyle='dotted')
        ax[0,0].set_title('Error in mean estimator')
        ax[0,1].violinplot([error_var_sf, error_var_mf_biased, error_var_mf_unbiased, error_var_mf_cv])
        ax[0,1].axhline(0, color='black', linestyle='dotted')
        ax[0,1].set_title('Error in variance estimator')
        ax[0,2].violinplot([error_sobol_sf[:, 0], error_sobol_mf_biased[:, 0], error_sobol_mf_unbiased[:, 0], error_sobol_mf_cv[:, 0]])
        ax[0,2].axhline(0, color='black', linestyle='dotted')
        ax[0,2].set_title('Error in Sobol estimator for $X_1$')
        ax[1,0].violinplot([error_sobol_sf[:, 1], error_sobol_mf_biased[:, 1], error_sobol_mf_unbiased[:, 1], error_sobol_mf_cv[:, 1]])
        ax[1,0].axhline(0, color='black', linestyle='dotted')
        ax[1,0].set_title('Error in Sobol estimator for $X_2$')
        ax[1,1].violinplot([error_sobol_sf[:, 2], error_sobol_mf_biased[:, 2], error_sobol_mf_unbiased[:, 2], error_sobol_mf_cv[:, 2]])
        ax[1,1].axhline(0, color='black', linestyle='dotted')
        ax[1,1].set_title('Error in Sobol estimator for $X_3$')
        plt.yscale('symlog')
        plt.show()
    
    else:
        print('No plot')

if __name__ == '__main__':
    statistical_study(500, type_plot='boxplot')
    # statistical_study(500, type_plot='violinplot')
    # statistical_study(500, type_plot='none')