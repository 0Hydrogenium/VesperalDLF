from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class WeibullMappingDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_name, timestep, predict_range, k, cumulative_prob, degradation_factor, initial_baseline, baseline_growth):
        self.x_list = []
        self.y_list = []
        self.mapped_y_list = []
        self.weibull_fault_indices = None
        self.weibull_baselines = None
        self.weibull_etas = None
        self.data_y = None  # 用于可视化
        self.weibull_data_y = None  # 用于可视化

        data_x = data.iloc[:, :].to_numpy()
        data_y = data.loc[:, target_name].to_numpy()

        weibull_data_y, weibull_fault_indices, weibull_baselines, weibull_etas = self.dynamic_weibull_cdf_mapping(data_y, k, cumulative_prob, degradation_factor, initial_baseline, baseline_growth)
        # 去除最后一个故障之后的正常数据
        data_x = data_x[: len(weibull_data_y), :]
        data_y = data_y[: len(weibull_data_y)]

        self.weibull_fault_indices = weibull_fault_indices
        self.weibull_baselines = weibull_baselines
        self.weibull_etas = weibull_etas
        self.data_y = data_y
        self.weibull_data_y = weibull_data_y

        # 转换为时间窗口
        for idx in range(len(data_x) - timestep - predict_range + 1):
            self.x_list.append(torch.tensor(data_x[idx: idx + timestep], dtype=torch.float32))
            self.y_list.append(torch.tensor(data_y[idx + timestep + predict_range - 1], dtype=torch.float32))
            self.mapped_y_list.append(torch.tensor(weibull_data_y[idx + timestep + predict_range - 1], dtype=torch.float32))

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, item):
        return self.x_list[item], self.y_list[item], self.mapped_y_list[item]

    def dynamic_weibull_cdf_mapping(self, data, k=2, cumulative_prob=0.99, degradation_factor=0.85, initial_baseline=0.0, baseline_growth=0.05):
        """

        :param data: 0/1序列
        :param k: Weibull分布形状参数 (k>1)
        :param cumulative_prob: 故障点目标累积概率
        :param degradation_factor: 设备状态退化因子 (0.8-0.95)
        :param initial_baseline: 初始基线故障概率 (0-1)
        :param baseline_growth: 每次维修后基线故障的增幅 (0-0.2)
        :return: 故障概率序列 (0-1), 故障点位置, 各段基线故障概率, 各段特征寿命
        """

        fault_indices = np.where(data == 1)[0]

        # 处理无故障情况
        if len(fault_indices) == 0:
            return np.zeros_like(data), fault_indices, [], []

        # 截断到最后一个故障点
        last_fault_idx = fault_indices[-1]
        data = data[:last_fault_idx + 1]
        fault_indices = fault_indices[fault_indices <= last_fault_idx]

        # 初始化结果数组
        mapped_data = np.zeros(len(data), dtype=float)
        baselines = []  # 存储各段起始基线故障概率
        etas = []  # 存储各段特征寿命

        # 计算基数
        log_base = - np.log(1 - cumulative_prob)
        # 当前基线概率（随时间增加）
        current_baseline = initial_baseline
        # 处理所有故障段
        for i, fault_idx in enumerate(fault_indices):
            # 确定当前段起点
            if i == 0:
                start_idx = 0
            else:
                start_idx = fault_indices[i - 1] + 1

            # 记录当前段基线值
            baselines.append(current_baseline)
            # 计算当前段运行时间
            operation_time = fault_idx - start_idx

            # 计算当前段特征寿命
            if operation_time == 0:
                # 特征寿命极短 (快速失效)
                eta = 0.1
            else:
                # 计算基础特征寿命
                base_eta = operation_time / (log_base ** (1 / k))
                # 应用退化效应: 随着故障次数增加，特征寿命缩短
                eta = base_eta * (degradation_factor ** i)
                # 考虑基线的影响 (基线越高，特征寿命越短)
                if current_baseline > 0.3:
                    eta *= (1 - current_baseline)

            etas.append(eta)

            # 计算当前段累积分布函数CDF (从当前基线值开始增长)
            t_segment = np.arange(operation_time + 1)
            # 计算Weibull分布的值 (从0到1)
            weibull_cdf = 1 - np.exp(- (t_segment / eta) ** k)
            # 从基线值开始增长到目标累积概率
            adjusted_cdf = current_baseline + (cumulative_prob - current_baseline) * weibull_cdf
            # 更新结果 (仅覆盖当前段)
            mapped_data[start_idx: start_idx + len(adjusted_cdf)] = adjusted_cdf
            # 更新基线概率: 每次维修后基线增加 (设备整体状态恶化)
            # current_baseline += baseline_growth
            current_baseline *= (1 + baseline_growth)
            # current_baseline = min(current_baseline, 0.5)  # 上限控制

        # 确保段内单调递增 (但允许段间下降)
        for i, fault_idx in enumerate(fault_indices):
            if i == 0:
                start_idx = 0
            else:
                start_idx = fault_indices[i - 1] + 1

            segment = mapped_data[start_idx: fault_idx + 1]

            # 仅保证段内单调递增
            mapped_data[start_idx: fault_idx + 1] = np.maximum.accumulate(segment)

        return mapped_data, fault_indices, baselines, etas

