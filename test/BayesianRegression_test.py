import numpy as np

from module.model.BayesianRegression import BayesianRegression
from module.model.KernelBayesianRegression import KernelBayesianRegression

if __name__ == '__main__':
    X = np.random.randn(10, 5) @ np.random.randn(5, 5)

    print("KernelBayesianRegression result:")
    kernel_model = KernelBayesianRegression()
    kernel_model.experience_params_setting(X)
    print(f"before_nll: {kernel_model.compute_baseline_nll(X)}")
    kernel_model.train(X)
    print(f"after_nll: {kernel_model.compute_negative_log_likelihood(X, X)}")

    print("BayesianRegression result:")
    model = BayesianRegression()
    model.experience_params_setting(X)
    print(f"before_nll: {model.compute_baseline_nll(X)}")
    model.train(X)
    print(f"after_nll: {model.compute_negative_log_likelihood(X, X)}")
