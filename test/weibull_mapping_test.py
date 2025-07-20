import numpy as np
import matplotlib.pyplot as plt


def dynamic_weibull_cdf_mapping(data, k=2, cumulative_prob=0.99, degradation_factor=0.85,
                                initial_baseline=0.0, baseline_growth=0.05):
    """
    修正的动态Weibull模型：更准确反映维修后的状态重置

    参数:
        data: 0/1序列（0-正常，1-故障）
        k: Weibull形状参数（>1，默认2）
        cumulative_prob: 故障点目标累积概率（默认0.99）
        degradation_factor: 设备状态退化因子（0.8-0.95，默认0.85）
        initial_baseline: 初始基线故障概率（0-1，默认0.0）
        baseline_growth: 每次维修后基线故障概率的增幅（0-0.2，默认0.05）

    返回:
        result: 故障概率序列（0-1）
        fault_indices: 故障点位置
        baselines: 各段基线故障概率
        etas: 各段特征寿命
    """
    # 参数验证
    if k <= 1:
        raise ValueError("形状参数k必须>1")
    if not 0.7 <= degradation_factor <= 0.95:
        raise ValueError("退化因子应在0.7-0.95之间")
    if not 0 <= initial_baseline <= 0.3:
        raise ValueError("初始基线值应在0-0.3之间")
    if not 0 <= baseline_growth <= 0.2:
        raise ValueError("基线增幅应在0-0.2之间")

    data = np.array(data)
    fault_indices = np.where(data == 1)[0]

    # 处理无故障情况
    if len(fault_indices) == 0:
        return np.zeros_like(data), fault_indices, [], []

    # 截断到最后一个故障点
    last_fault_idx = fault_indices[-1]
    data = data[:last_fault_idx + 1]
    fault_indices = fault_indices[fault_indices <= last_fault_idx]

    # 初始化结果数组
    result = np.zeros(len(data), dtype=float)
    baselines = []  # 存储各段起始基线故障概率
    etas = []  # 存储各段特征寿命

    # 计算基数
    log_base = -np.log(1 - cumulative_prob)

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

        # 计算当前段特征寿命η
        if operation_time == 0:  # 连续故障
            # 特征寿命极短（快速失效）
            eta = 0.1
        else:
            # 计算基础特征寿命
            base_eta = operation_time / (log_base ** (1 / k))

            # 应用退化效应：随着故障次数增加，特征寿命缩短
            eta = base_eta * (degradation_factor ** i)

            # 考虑基线的影响（基线越高，特征寿命越短）
            if current_baseline > 0.3:
                eta *= (1 - current_baseline)

        etas.append(eta)

        # 计算当前段CDF（从当前基线值开始增长）
        t_segment = np.arange(operation_time + 1)

        # 计算Weibull分布的值（从0到1）
        weibull_cdf = 1 - np.exp(-(t_segment / eta) ** k)

        # 从基线值开始增长到目标累积概率
        adjusted_cdf = current_baseline + (cumulative_prob - current_baseline) * weibull_cdf

        # 更新结果（仅覆盖当前段）
        result[start_idx:start_idx + len(adjusted_cdf)] = adjusted_cdf

        # 更新基线概率：每次维修后基线增加（设备整体状态恶化）
        current_baseline += baseline_growth
        current_baseline = min(current_baseline, 0.5)  # 上限控制

    # 确保段内单调递增（但允许段间下降）
    for i, fault_idx in enumerate(fault_indices):
        if i == 0:
            start_idx = 0
        else:
            start_idx = fault_indices[i - 1] + 1

        segment = result[start_idx:fault_idx + 1]

        # 仅保证段内单调递增
        result[start_idx:fault_idx + 1] = np.maximum.accumulate(segment)

    return result, fault_indices, baselines, etas


# 增强的可视化函数
def plot_enhanced_weibull_result(data, mapped_data, fault_indices, baselines, etas):
    plt.figure(figsize=(14, 10))

    # 原始序列和分段标记
    plt.subplot(3, 1, 1)
    plt.stem(range(len(data)), data, linefmt='C0-', markerfmt='C1o', basefmt=" ")
    plt.title('原始序列和故障点位置')
    plt.ylabel('状态')
    plt.ylim(-0.1, 1.2)

    # 标记各段
    colors = plt.cm.viridis(np.linspace(0, 1, len(fault_indices)))
    for i, idx in enumerate(fault_indices):
        if i == 0:
            start = 0
        else:
            start = fault_indices[i - 1] + 1
        end = idx

        plt.axvspan(start, end, alpha=0.2, color=colors[i])
        plt.text((start + end) / 2, 0.5, f'段 {i + 1}\nη={etas[i]:.2f}\n基线={baselines[i]:.3f}',
                 ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    plt.grid(alpha=0.3)

    # 故障概率曲线
    plt.subplot(3, 1, 2)
    plt.plot(mapped_data, 'g-', linewidth=2, label='故障概率')
    plt.scatter(fault_indices, mapped_data[fault_indices], c='red', s=80,
                zorder=5, marker='X', label='故障点')

    # 标记基线
    for i, baseline in enumerate(baselines):
        if i == 0:
            start = 0
        else:
            start = fault_indices[i - 1] + 1

        plt.hlines(baseline, start, fault_indices[i], colors='purple', linestyles='dashed', alpha=0.7)

    # 标记特征寿命变化
    for i, (eta, baseline) in enumerate(zip(etas, baselines)):
        plt.annotate(f'η={eta:.2f}',
                     xy=(fault_indices[i], mapped_data[fault_indices[i]]),
                     xytext=(5, 5 if i % 2 == 0 else -25), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='gray'))

    plt.axhline(y=0.99, color='gray', linestyle='--', alpha=0.7)
    plt.title('动态Weibull映射结果')
    plt.xlabel('时间索引')
    plt.ylabel('故障概率')
    plt.legend()
    plt.grid(alpha=0.3)

    # 参数变化趋势
    plt.subplot(3, 1, 3)
    # 特征寿命η
    plt.plot(range(1, len(etas) + 1), etas, 'bo-', markersize=8, label='特征寿命η')
    # 基线值
    plt.plot(range(1, len(baselines) + 1), baselines, 'rs--', markersize=8, label='基线故障概率')

    plt.title('设备状态退化趋势')
    plt.xlabel('故障段序号')
    plt.ylabel('参数值')
    plt.xticks(range(1, len(etas) + 1))
    plt.grid(alpha=0.3)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('enhanced_weibull_mapping.png', dpi=150)
    plt.show()


# 示例使用
if __name__ == "__main__":
    # 创建含多次故障的序列
    input_data = [0] * 5 + [1] + [0] * 6 + [1] + [0] * 7 + [1] + [0] * 4 + [1] + [0] * 5

    # 执行动态Weibull映射
    # 参数说明：
    # degradation_factor=0.88: 每次故障后特征寿命衰减12%
    # initial_baseline=0.02: 初始基线故障概率为2%
    # baseline_growth=0.06: 每次维修后基线概率增加6%
    mapped_data, fault_indices, baselines, etas = dynamic_weibull_cdf_mapping(
        input_data,
        k=2.0,
        degradation_factor=0.88,
        initial_baseline=0.02,
        baseline_growth=0.06
    )

    # 打印结果
    print("故障点位置:", fault_indices)
    print("基线故障概率:", [round(b, 4) for b in baselines])
    print("特征寿命η:", [round(eta, 2) for eta in etas])
    print("\n映射结果:")
    for i in range(min(25, len(mapped_data))):  # 只打印前25个点
        print(f"时间点 {i}: {mapped_data[i]:.4f}")

    # 可视化结果
    plot_enhanced_weibull_result(input_data[:len(mapped_data)], mapped_data,
                                 fault_indices, baselines, etas)