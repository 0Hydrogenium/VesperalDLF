import numpy as np

from module.model.BayesianRegression import BayesianRegression


class KernelBayesianRegression(BayesianRegression):
    def __init__(self, alpha=1e-6, beta=1e6, kernel="rbf", length_scale=1):
        super().__init__(alpha, beta)
        self.kernel = kernel
        self.length_scale = length_scale
        self.X_train = None  # 训练数据
        self.K_inv = None  # 正则化核矩阵的逆
        self.dual_coeff = None  # 对偶系数

    def experience_params_setting(self, X):
        # 基于数据标准差的经验设置
        self.alpha = 1 / (10 * np.std(X) ** 2)  # 温和正则化
        self.beta = 1 / (0.1 * np.std(X) ** 2)  # 中等噪声假设
        self.length_scale = np.std(X)  # 长度尺度与数据标准差匹配

    def train(self, X):
        self.X_train = X
        # 计算核矩阵
        K = self.compute_kernel(X, X)
        # 正则化参数
        lambda_reg = self.alpha / self.beta
        K_reg = K + lambda_reg * np.eye(X.shape[0])
        # 使用Cholesky分解或SVD计算后验协方差的逆
        self.K_inv = self.matrix_decomposition(K_reg)
        # 计算对偶系数
        self.dual_coeff = self.K_inv @ X
        return self.predict(X)[0]

    def predict(self, X):
        k_star = self.compute_kernel(X, self.X_train)
        # 预测均值
        mean_pred = k_star @ self.dual_coeff
        # 预测方差
        if self.kernel == "rbf":
            k_star_star = self.compute_kernel(X, X).diagonal()  # 计算对角元素
        else:
            k_star_star = np.sum(X ** 2, axis=1)  # 其他核需要计算
        var_pred = k_star_star - np.sum((k_star @ self.K_inv) * k_star, axis=1) + 1 / self.beta
        # 确保方差非负
        var_pred = np.maximum(var_pred, 1e-8)
        return mean_pred, var_pred

    def compute_kernel(self, X1, X2):
        if self.kernel == "rbf":
            return self.rbf_kernel(X1, X2)

        raise ValueError("Unsupported kernel type")

    def rbf_kernel(self, X1, X2):
        """
        径向基函数（RBF）核

        :param X1: 第一个数据集
        :param X2: 第二个数据集
        :return: RBF核矩阵
        """

        # 计算平方欧式距离
        X1_norm = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
        X2_norm = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
        dist_sq = X1_norm - 2 * np.dot(X1, X2.T) + X2_norm
        # 数据稳定性处理
        dist_sq = np.maximum(dist_sq, 1e-12)
        return np.exp(- dist_sq / (2 * self.length_scale ** 2))
