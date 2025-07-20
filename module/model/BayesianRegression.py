import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class BayesianRegression:
    """
        compute_negative_log_likelihood出现Nan的原因：数值不稳定性，尤其是在指数和方差项计算中
        问题：
            1.方差项过大/过小：
                - 当预测方差var_pred非常小接近零时，计算1/var_pred会非常大
                - 当方差非常大时，SSE/var_pred可能趋近于零导致信息丢失
            2.指数运算溢出：
                - 当(residuals**2)/var_pred值非常大时，指数计算会接近机器精度极限
            3.残差放大效应：
                - 在多元重构任务中，SSE会随维度增加而变大，加剧数值不稳定
    """

    def __init__(self, alpha=1e-6, beta=1e6):
        self.alpha = alpha  # 先验精度
        self.beta = beta  # 噪声精度
        self.m_N = None  # 后验均值
        self.S_N = None  # 后验协方差的逆

    def experience_params_setting(self, X):
        # 基于数据标准差的经验设置
        data_std = np.std(X)
        y_std = np.std(X)
        alpha_0 = 1 / (10 * data_std ** 2)  # 温和正则化
        beta_0 = 1 / (0.1 * y_std ** 2)  # 中等噪声假设
        self.alpha = alpha_0
        self.beta = beta_0

    def train(self, X):
        # 构建设计矩阵，添加偏置项和特征交叉项，增强模型表达能力
        X_design = self.build_design_matrix(X)
        # 计算后验协方差矩阵
        posterior_cov = self.compute_posterior_cov(X_design)
        # 使用Cholesky分解或SVD计算后验协方差的逆
        self.S_N = self.matrix_decomposition(posterior_cov)
        # 计算后验均值
        self.m_N = self.beta * self.S_N @ X_design.T @ X
        return self.predict(X)[0]  # 返回预测均值

    def test(self, X):
        return self.predict(X)[0]  # 返回预测均值

    def predict(self, X):
        # 设计矩阵拓展
        X_design = self.build_design_matrix(X)
        # 预测均值
        mean_pred = X_design @ self.m_N
        # 预测方差
        var_pred = 1 / self.beta + np.sum((X_design @ self.S_N) * X_design, axis=1)
        # 确保方差非负
        var_pred = np.maximum(var_pred, 1e-8)
        return mean_pred, var_pred

    def compute_negative_log_likelihood(self, X, y):
        # 计算负对数似然NLL
        mean_pred, var_pred = self.predict(X)
        n, d = y.shape
        # 计算残差平方和（每个样本所有特征维度的平方和）
        residuals = y - mean_pred
        squared_residuals = np.sum(residuals ** 2, axis=1) / var_pred
        # # 拓展方差维度以匹配残差形状
        # var_pred = var_pred[:, np.newaxis]
        # 稳定计算每个样本的对数似然
        ll_per_sample = -0.5 * (d * (np.log(2 * np.pi) + np.log(var_pred)) + squared_residuals)
        nll = - np.sum(ll_per_sample) / n
        return nll

    def compute_baseline_nll(self, X):
        """
        使用高斯模型作为基准，计算基准负对数似然值NLL

        :param X:
        :return:
        """

        n, d = X.shape
        # 计算每个特征的均值和方差
        mean_per_feature = np.mean(X, axis=0)
        var_per_feature = np.var(X, axis=0, ddof=0)  # 使用最大似然估计
        var_per_feature = np.maximum(var_per_feature, 1e-8)

        # 计算每个数据点在基准模型下的对数似然
        ll = 0
        for i in range(n):
            for j in range(d):
                # 高斯分布的对数似然
                residuals = X[i, j] - mean_per_feature[j]
                log_var = np.log(var_per_feature[j])
                ll -= 0.5 * (np.log(2 * np.pi) + log_var + residuals ** 2 / var_per_feature[j])

        return - ll / n

    def matrix_decomposition(self, matrix):
        """
        使用Cholesky分解计算矩阵逆

        :param matrix:
        :return:
        """

        try:
            # 使用Cholesky分解提高数值稳定性
            L = np.linalg.cholesky(matrix)
            return np.linalg.inv(L.T) @ np.linalg.inv(L)
        except np.linalg.LinAlgError:
            print("BayesianRegression rolllback to direct inv")
            return np.linalg.inv(matrix)

    def compute_posterior_cov(self, X_design):
        """
        计算后验协方差矩阵

        :param X_design: 设计矩阵
        :return: 后验协方差矩阵
        """

        I = np.eye(X_design.shape[1])
        # 添加正则项确保数值稳定
        ridge = 1e-6 * np.trace(X_design.T @ X_design) / X_design.shape[1] * I
        cov = self.alpha * I + self.beta * (X_design.T @ X_design) + ridge
        return cov

    def build_design_matrix(self, X):
        """
        构建设计矩阵（原始特征+偏置项+特征交叉项）

        :param X: 输入数据 (n_samples, n_features)
        :return: 设计矩阵 (n_samples, n_features + 2)
        """

        # ones = np.ones((X.shape[0], 1))
        # cross_term = np.prod(X, axis=1, keepdims=True)
        # return np.hstack([X, ones, cross_term])

        # 用交互特征代替全乘积
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        return poly.fit_transform(X)
