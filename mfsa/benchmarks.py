import numpy as np
from abc import abstractmethod
import json
from mfsa.helper import variance_of_estimators_sf, variance_of_estimators_mfts, variance_of_estimators_mfcv
import matplotlib.pyplot as plt


class BenchmarkAbstract():

    def __init__(self, dim) -> None:
        self.dim = dim

    @abstractmethod
    def hf(self, x):
        pass

    @abstractmethod
    def lf(self, x):
        pass

    @abstractmethod
    def calculate_statistics(self):
        pass

    @abstractmethod
    def get_info(self):
        pass

    def save_info(self, path):
        info = self.get_info()
        with open(path+'info.json', 'w+') as f:
            json.dump(info, f)

    def estimate_mean_variance(self, num_samples):
        samples = np.random.uniform(0,1, (num_samples, self.dim))
        Y = self.hf(samples)
        return np.mean(Y), np.var(Y, ddof=1)
    
    def estimate_lf_hf_covariance(self, num_samples):
        samples = np.random.uniform(0,1, (num_samples, self.dim))
        Yhf = self.hf(samples)
        Ylf = self.lf(samples)
        return np.sum((Yhf - np.mean(Yhf))*(Ylf - np.mean(Ylf))) / (num_samples - 1)
    
    def estimate_lf_mean_variance(self, num_samples):
        samples = np.random.uniform(0,1, (num_samples, self.dim))
        Y = self.lf(samples)
        return np.mean(Y), np.var(Y, ddof=1)

    def get_f_list(self):
            return [self.lf, self.hf]
    
    def get_lf_difference_stats(self, num_samples=1000):
        samples = np.random.uniform(0,1, (num_samples, self.dim))
        Yhf = self.hf(samples)
        Ylf = self.lf(samples)
        diff = Yhf - Ylf
        mean_diff = np.mean(diff)
        variance_diff = np.var(diff, ddof=1)
        return np.mean(Ylf), np.var(Ylf, ddof=1), mean_diff, variance_diff
    

class Gfunction(BenchmarkAbstract):

    def __init__(self, dim, a) -> None:
        super().__init__(dim)
        self.a = np.array(a)
        assert len(self.a) == self.dim, "The length of a must be equal to the dimension of the problem."
        assert (self.a != -1).any(), "The a values must be different from -1."
        self.constant = 1/(3*(1+self.a)**2)
        self.a_lf, self.shift = self.a + 1, 0.2
        self.constant_lf = 1/(3*(1+self.a_lf)**2)

    def hf(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            x = x.T
        return np.prod((np.abs(4*x - 2) + self.a) / (1 + self.a), axis=1)
    
    def lf(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            x = x.T
        return np.prod((np.abs(4*x - 2) + self.a_lf) / (1 + self.a_lf), axis=1) + self.shift
    
    def lf_statistics(self):
        mean = self.shift + 1
        variance = np.prod(1+ self.constant_lf) - 1
        first_order_sobol =  self.constant_lf / variance
        total_order_sobol = self.constant_lf * (1+ self.constant_lf) * np.prod(1+ self.constant_lf) / variance
        return mean, variance, first_order_sobol, total_order_sobol
    
    def calculate_statistics(self):
        mean = 1
        variance = np.prod(1+ self.constant) - 1
        first_order_sobol =  self.constant / variance
        total_order_sobol = self.constant * (1+ self.constant) * np.prod(1+ self.constant) / variance
        return mean, variance, first_order_sobol, total_order_sobol
    
    def get_info(self):
        return {'name': 'G-function', 'dim': self.dim, 'a': self.a.tolist()}
    

class Gfunction1(BenchmarkAbstract):

    def __init__(self, dim, a) -> None:
        super().__init__(dim)
        self.a = np.array(a)
        assert len(self.a) == self.dim, "The length of a must be equal to the dimension of the problem."
        assert (self.a != -1).any(), "The a values must be different from -1."
        self.constant = 1/(3*(1+self.a)**2)

    def hf(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            x = x.T
        return np.prod((np.abs(4*x - 2) + self.a) / (1 + self.a), axis=1)
    
    def lf(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            x = x.T
        return np.prod((np.abs(4*x - 2.5) + self.a) / (1 + self.a), axis=1)
    
    def calculate_statistics(self):
        mean = 1
        variance = np.prod(1+ self.constant) - 1
        first_order_sobol =  self.constant / variance
        total_order_sobol = self.constant * (1+ self.constant) * np.prod(1+ self.constant) / variance
        return mean, variance, first_order_sobol, total_order_sobol
    
    def get_info(self):
        return {'name': 'G-function-lf-modified', 'dim': self.dim, 'a': self.a.tolist()}

    
class Gstarfunction(BenchmarkAbstract):

    def __init__(self, dim, a, alpha=0.75, delta=0.3) -> None:
        super().__init__(dim)
        self.a = np.array(a)
        assert len(self.a) == self.dim, "The length of a must be equal to the dimension of the problem."
        assert (self.a != -1).any(), "The a values must be different from -1."
        assert alpha >= 0, "The alpha value must be greater than 0."
        assert delta <= 1 and delta >= 0, "The delta value must be between 0 and 1."
        self.alpha = alpha * np.ones(self.dim)
        self.delta = delta * np.ones(self.dim)
        self.constant = self.alpha**2 / ((1 + 2*self.alpha)*(1 + self.a)**2)
        self.alpha_lf, self.delta_lf, self.a_lf, self.shift = self.alpha+0.05, self.delta, self.a + 1, 0.2
        self.constant_lf = self.alpha_lf**2 / ((1 + 2*self.alpha_lf)*(1 + self.a_lf)**2)

    def hf(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            x = x.T
        return np.prod(((1 + self.alpha) * np.power(np.abs(2*(x + self.delta - np.array(x+self.delta, dtype=np.int32)) - 1), self.alpha) + self.a) / (1 + self.a), axis=1)
    
    def lf(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            x = x.T
        return np.prod(((1 + self.alpha_lf) * np.power(np.abs(2*(x + self.delta_lf - np.array(x+self.delta_lf, dtype=np.int32)) - 1), self.alpha_lf) + self.a_lf) / (1 + self.a_lf), axis=1) + self.shift
    
    def lf_statistics(self):
        mean = self.shift + 1
        variance = np.prod(1+ self.constant_lf) - 1
        first_order_sobol =  self.constant_lf / variance
        total_order_sobol = self.constant_lf * (1+ self.constant_lf) * np.prod(1+ self.constant_lf) / variance
        return mean, variance, first_order_sobol, total_order_sobol
    
    def calculate_statistics(self):
        mean = 1
        variance = np.prod(1+ self.constant) - 1
        first_order_sobol =  self.constant / variance
        total_order_sobol = self.constant * (1+ self.constant) * np.prod(1+ self.constant) / variance
        return mean, variance, first_order_sobol, total_order_sobol
    
    def get_info(self):
        return {'name': 'G*-function', 'dim': self.dim, 'a': self.a.tolist(), 'alpha': self.alpha.tolist(), 'delta': self.delta.tolist()}
    
class C1(BenchmarkAbstract):

    def __init__(self, dim) -> None:
        super().__init__(dim) 

    def hf(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            x = x.T
        return 2**self.dim * np.prod(x, axis=1)
    
    def lf(self, x):
        x = np.atleast_2d(x)
        if x.shape[1] != self.dim:
            x = x.T
        return 1.9**self.dim * np.prod(x + 0.1, axis=1)
    
    def calculate_statistics(self):
        mean = 1
        variance = (4**self.dim) / (3**self.dim) - 1
        first_order_sobol = 1 / (3*variance) * np.ones(self.dim)
        total_order_sobol = self.dim * first_order_sobol / ((4/3)**(1-self.dim))
        return mean, variance, first_order_sobol, total_order_sobol
    
    def get_info(self):
        return {'name': 'C1-function', 'dim': self.dim}
    
class C2(BenchmarkAbstract):
    
        def __init__(self, dim) -> None:
            super().__init__(dim) 
    
        def hf(self, x):
            x = np.atleast_2d(x)
            if x.shape[1] != self.dim:
                x = x.T
            return np.prod(np.abs(4*x -2), axis=1)
        
        def lf(self, x):
            x = np.atleast_2d(x)
            if x.shape[1] != self.dim:
                x = x.T
            return np.prod(np.abs(4.2*x -2.1), axis=1)
        
        def calculate_statistics(self):
            mean = 1
            variance = (4**self.dim) / (3**self.dim) - 1
            first_order_sobol = 1 / (3*variance) * np.ones(self.dim)
            total_order_sobol = self.dim * first_order_sobol / ((4/3)**(1-self.dim))
            return mean, variance, first_order_sobol, total_order_sobol
        
        def get_info(self):
            return {'name': 'C2-function', 'dim': self.dim}
        
class Ishigami(BenchmarkAbstract):

    def __init__(self, a, b, lower=-np.pi, upper=np.pi):
        super().__init__(3) 
        self.lower, self.upper = lower, upper
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

    def lf(self, x):
        temp = np.atleast_2d(x)
        if temp.shape[1] != self.dim:
            temp = temp.T
        q = self.transform_coordinates(temp)
        a, b = self.a + 0.1, self.b + 0.005
        f =( np.sin(q[:, 0]) + a * (np.sin(q[:, 1])**2) + (b * np.sin(q[:, 0]) * (q[:, 2]**4)) ) 
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

    def get_info(self):
        return {'name': 'Ishigami-function', 'a': self.a, 'b': self.b, 'dim': self.dim}

# class Kfunction(BenchmarkAbstract):

#     def __init__(self) -> None:
#         dim = 10
#         super().__init__(dim)

#     def hf(self, x):
#         x = np.atleast_2d(x)
#         if x.shape[1] != self.dim:
#             x = x.T
#         return np.sum(np.power(-1, np.arange(1, self.dim+1)) * np.cumprod(x,axis=1), axis=1)
    
#     def lf(self, x):
#         x = np.atleast_2d(x)
#         if x.shape[1] != self.dim:
#             x = x.T
#         return np.sum(1.1*np.power(-1, np.arange(1, self.dim+1)) * np.cumprod(x,axis=1) + 0.2, axis=1)
    
#     def calculate_statistics(self):
#         variance = (1/10)*(1/3)**self.dim + (1/18) - (1/9)*(1/2)**(2*self.dim) + ((-1)**(self.dim+1))*(2/45)*(1/2)**self.dim
#         return 0, variance, 0, 0

def Gstar_delta_study():
    dim = 1
    # deltas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9, 1.0]
    deltas = [0.2,0.21]
    x = np.linspace(0, 1, 100)
    benchmarks = []
    benchmarks.append(Gstarfunction(dim, np.arange(1, dim+1), alpha=0.75, delta=0.1))
    benchmarks.append(Gstarfunction(dim, np.arange(1, dim+1)+1, alpha=0.76, delta=0.11))
    # for delta in deltas:
        # benchmarks.append(Gstarfunction(dim, np.arange(1, dim+1), delta=delta))
    for benchmark in benchmarks:
        plt.plot(x, benchmark.hf(x), label=f'delta={benchmark.delta}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Gstar_delta_study()
    dim = 4
    benchmark = Gstarfunction(dim, np.arange(1, dim+1), alpha=0.2)
    # benchmark = Ishigami(1, 0.01)
    # cov = benchmark.estimate_lf_hf_covariance(100000)
    hf_stats = benchmark.calculate_statistics()
    print(hf_stats)
    # print(variance_of_estimators_mfts(benchmark.lf, benchmark.hf, [334,23], benchmark.dim, num_samples=50))
    # lf_stats = benchmark.estimate_lf_mean_variance(100000)
    # print(cov, hf_stats[0], hf_stats[1], lf_stats)
    # p = cov/np.sqrt(lf_stats[1]*hf_stats[1])
    # ch = 5
    # constant =  a**2 + (a**2/ch) - (2*p*a)
    # print(p,constant)
    # np.random.seed(0)
    # print("MFTS", variance_of_estimators_mfts(benchmark.lf, benchmark.hf, [334,23], benchmark.dim))
    # np.random.seed(0)
    # print("SF",variance_of_estimators_sf(benchmark.hf, 100, benchmark.dim))
    # np.random.seed(0)
    # print("MFCV",variance_of_estimators_mfcv(benchmark.lf, benchmark.hf, [334,23], benchmark.dim))