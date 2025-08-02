import mesa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import random
import networkx as nx


class State(Enum):
    SUSCEPTIBLE = 0  # 易感状态
    INFECTED = 1  # 感染状态
    RECOVERED = 2  # 恢复状态
    REMOVED = 3  # 被移除状态


class UserAgent(mesa.Agent):
    """用户Agent"""

    def __init__(self, unique_id, model, influence, followers, credibility):
        # 使用新的初始化方式
        super().__init__(model)
        self.state = State.SUSCEPTIBLE
        self.influence = influence  # 影响力基数(1-10)
        self.followers = followers  # 粉丝数
        self.credibility = credibility  # 可信度(0-1)
        self.infected_time = None  # 感染时间
        self.posts = []  # 发布的博客列表
        self.engagement = 0  # 参与度

    def step(self):
        # 状态转换逻辑
        if self.state == State.SUSCEPTIBLE:
            self._try_get_infected()
        elif self.state == State.INFECTED:
            self._try_recover()
            self._spread_influence()
            # 检测是否触发平台干预
            if self.model.detect_intervention_condition(self):
                self.state = State.REMOVED
                self.model.removed_count += 1
                self.model.infected_count -= 1
        elif self.state == State.RECOVERED:
            self._lose_influence()

    def _try_get_infected(self):
        # 被邻居感染的概率
        infected_neighbors = [
            n for n in self.model.grid.get_neighbors(self.pos, include_center=False)
            if n.state == State.INFECTED
        ]
        if not infected_neighbors:
            return

        # 计算感染概率 (与邻居影响力和粉丝数相关)
        total_risk = sum(n.influence * np.log1p(n.followers) for n in infected_neighbors)
        infection_prob = 1 - np.exp(-self.model.beta * total_risk)


        # 用户可信度影响感染概率
        infection_prob *= (1 - self.credibility)

        if self.model.random.random() < infection_prob:  # 使用model的random
            self.state = State.INFECTED
            self.infected_time = self.model.schedule.time
            self.model.infected_count += 1
            self.model.susceptible_count -= 1

    def _try_recover(self):
        # 自然恢复概率
        recovery_prob = self.model.gamma * (1 + self.credibility)
        if self.model.random.random() < recovery_prob:  # 使用model的random
            self.state = State.RECOVERED
            self.model.infected_count -= 1
            self.model.recovered_count += 1

    def _spread_influence(self):
        # 传播舆情（创建新博客）
        if self.model.random.random() < 0.3:  # 30%概率发布新博客，使用model的random
            sentiment = random.choice(["negative", "highly_negative"])
            new_post = {
                'id': f"post_{self.unique_id}_{len(self.posts)}",
                'time_series': self.model.schedule.time,
                'sentiment': sentiment,
                'base_influence': self.influence,
                'current_influence': self.influence,
                'velocity': 0,
                'acceleration': 0,
                'shares': 0,
                'comments': 0,
                'likes': 0
            }
            self.posts.append(new_post)

        # 更新已有博客的影响力动态
        for post in self.posts:
            # 影响力自然增长（随机）
            delta = self.model.random.uniform(0.1, 0.5) * self.influence  # 使用model的random

            # 情感强度影响传播
            if post['sentiment'] == "highly_negative":
                delta *= 1.5

            post['current_influence'] += delta
            post['shares'] += int(delta * 0.5)
            post['likes'] += int(delta * 2)
            post['comments'] += int(delta * 0.8)

            # 计算传播动力学指标
            prev_v = post.get('velocity', 0)
            post['velocity'] = delta / self.model.time_step
            post['acceleration'] = (post['velocity'] - prev_v) / self.model.time_step

    def _lose_influence(self):
        # 影响力随时间衰减
        for post in self.posts:
            decay_factor = 1 - self.model.alpha * (1 + self.credibility)
            post['current_influence'] *= decay_factor
            post['velocity'] *= decay_factor
            post['acceleration'] *= decay_factor


class RumorSpreadModel(mesa.Model):
    """舆情传播模型"""

    def __init__(self, N, width, height, beta=0.02, gamma=0.05, alpha=0.01,
                 intervention_threshold=2.0, removal_rate=0.2, network_type="random", seed=None):
        super().__init__()
        # 初始化随机数生成器
        self.random = np.random.default_rng(seed)
        self.num_agents = N
        self.grid = self._create_network(width, height, network_type)

        # 使用新的调度器API
        self.schedule = mesa.time.RandomActivationByType(self)
        self.schedule_types = {"user": UserAgent}  # 定义Agent类型

        self.beta = beta  # 感染率
        self.gamma = gamma  # 恢复率
        self.alpha = alpha  # 影响力衰减系数
        self.intervention_threshold = intervention_threshold  # 干预阈值（加速度）
        self.removal_rate = removal_rate  # 移除率（干预强度）
        self.time_step = 1.0  # 时间步长（小时）
        self.network_type = network_type

        # 状态计数器
        self.susceptible_count = 0
        self.infected_count = 0
        self.recovered_count = 0
        self.removed_count = 0

        # 创建Agent
        for i in range(self.num_agents):
            # 随机生成属性
            influence = self.random.integers(1, 10)  # 使用model的random
            followers = self.random.integers(100, 10000)  # 使用model的random
            credibility = self.random.uniform(0.1, 0.9)  # 使用model的random

            agent = UserAgent(i, self, influence, followers, credibility)
            self.schedule.add(agent)

            # 放置Agent
            if network_type == "random":
                x = self.random.integers(0, self.grid.width)
                y = self.random.integers(0, self.grid.height)
                self.grid.place_agent(agent, (x, y))
            else:  # 对于小世界网络，位置已确定
                self.grid.place_agent(agent, i)

        # 初始化感染源
        patient_zero = self.random.choice(self.schedule.agents)
        patient_zero.state = State.INFECTED
        patient_zero.infected_time = 0
        self.infected_count = 1
        self.susceptible_count = self.num_agents - 1

        # 数据收集器
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Susceptible": "susceptible_count",
                "Infected": "infected_count",
                "Recovered": "recovered_count",
                "Removed": "removed_count",
                "Total_Influence": self.calculate_total_influence
            },
            agent_reporters={
                "State": lambda a: a.state.name,
                "Influence": "influence",
                "Followers": "followers"
            }
        )

    def _create_network(self, width, height, network_type):
        """创建不同类型的社交网络"""
        if network_type == "random":
            return mesa.space.MultiGrid(width, height, True)
        elif network_type == "small_world":
            # 创建小世界网络
            graph = nx.watts_strogatz_graph(self.num_agents, k=4, p=0.3, seed=self.random)
            return mesa.space.NetworkGrid(graph)
        else:
            raise ValueError(f"未知网络类型: {network_type}")

    def step(self):
        """推进模型一个时间步"""
        self.datacollector.collect(self)
        self.schedule.step()

    def calculate_total_influence(self):
        """计算系统中所有感染节点的总影响力"""
        total = 0
        for agent in self.schedule.agents:
            if agent.state == State.INFECTED:
                for post in agent.posts:
                    total += post['current_influence']
        return total

    def detect_intervention_condition(self, agent):
        """
        检测是否满足干预条件（基于传播加速度）

        参数:
            agent: 待检测的Agent

        返回:
            bool: 是否执行移除
        """
        if not agent.posts:
            return False

        # 检查最新博客的加速度
        latest_post = agent.posts[-1]
        if latest_post['acceleration'] > self.intervention_threshold:
            # 以一定概率移除（模拟平台审核效率）
            return self.random.random() < self.removal_rate  # 使用model的random
        return False


def run_simulation(params, title):
    """运行模拟并绘制结果"""
    model = RumorSpreadModel(**params)

    # 运行模型
    for i in range(200):
        model.step()

    # 获取数据
    data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()

    # 绘制状态变化
    plt.figure(figsize=(14, 10))

    # 状态变化图
    plt.subplot(2, 2, 1)
    plt.plot(data.index, data.Susceptible, label='易感节点', color='blue')
    plt.plot(data.index, data.Infected, label='感染节点', color='red')
    plt.plot(data.index, data.Recovered, label='恢复节点', color='green')
    plt.plot(data.index, data.Removed, label='移除节点', color='black')
    plt.title(f'{title} - 节点状态变化', fontsize=14)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('节点数量', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    # 总影响力变化
    plt.subplot(2, 2, 2)
    plt.plot(data.index, data.Total_Influence, color='purple')
    plt.title('负面舆情总影响力变化', fontsize=14)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('总影响力', fontsize=12)
    plt.grid(alpha=0.3)

    # 状态分布饼图（最终状态）
    plt.subplot(2, 2, 3)
    final_counts = data.iloc[-1][['Susceptible', 'Infected', 'Recovered', 'Removed']]
    labels = ['易感节点', '感染节点', '恢复节点', '移除节点']
    colors = ['blue', 'red', 'green', 'black']
    plt.pie(final_counts, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('最终状态分布', fontsize=14)

    # 峰值时间分析
    peak_time = data.Infected.idxmax()
    peak_value = data.Infected.max()

    # 平息时间（感染节点<5%）
    decay_time = data[data.Infected < 0.05 * params['N']].index.min()
    if pd.isna(decay_time):
        decay_time = "未平息"

    # 添加统计信息
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = (
        f"网络类型: {params['network_type']}\n"
        f"干预阈值: {params['intervention_threshold']}\n"
        f"移除率: {params['removal_rate']}\n"
        f"峰值时间: {peak_time}\n"
        f"峰值感染数: {peak_value}\n"
        f"平息时间: {decay_time}\n"
        f"总移除数: {data.Removed.iloc[-1]}"
    )
    plt.text(0.1, 0.5, stats_text, fontsize=12,
             bbox=dict(facecolor='lightgray', alpha=0.5))

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"{title.replace(' ', '_')}.png")  # 保存图像
    plt.show()

    return data, agent_data


# 实验配置（添加seed参数确保可重复性）
experiments = {
    "无干预": {
        "N": 200,  # 减少节点数加速运行
        "width": 30,
        "height": 30,
        "beta": 0.03,
        "gamma": 0.02,
        "intervention_threshold": 100,  # 极高阈值，相当于无干预
        "removal_rate": 0.0,
        "network_type": "random",
        "seed": 42
    },
    "精准干预": {
        "N": 200,
        "width": 30,
        "height": 30,
        "beta": 0.03,
        "gamma": 0.02,
        "intervention_threshold": 1.5,
        "removal_rate": 0.7,
        "network_type": "random",
        "seed": 42
    },
    "随机干预": {
        "N": 200,
        "width": 30,
        "height": 30,
        "beta": 0.03,
        "gamma": 0.02,
        "intervention_threshold": 100,
        "removal_rate": 0.1,  # 随机移除感染节点
        "network_type": "random",
        "seed": 42
    },
    "小世界网络干预": {
        "N": 200,
        "width": 30,
        "height": 30,
        "beta": 0.03,
        "gamma": 0.02,
        "intervention_threshold": 1.5,
        "removal_rate": 0.7,
        "network_type": "small_world",
        "seed": 42
    }
}

# 运行所有实验
results = {}
for name, params in experiments.items():
    print(f"正在运行实验: {name}")
    data, agent_data = run_simulation(params, name)
    results[name] = {
        "model_data": data,
        "agent_data": agent_data
    }
    print(f"实验 {name} 完成\n{'=' * 50}")


# 比较不同策略效果
def compare_strategies(results):
    """比较不同干预策略效果"""
    metrics = {}

    for strategy, data in results.items():
        df = data['model_data']
        # 获取Agent数量（N）
        agent_count = results[strategy]['agent_data'].index.get_level_values('AgentID').nunique()

        # 计算平息时间（感染节点<5%）
        decay_condition = df.Infected < 0.05 * agent_count
        decay_time = df[decay_condition].index.min() if decay_condition.any() else len(df)

        metrics[strategy] = {
            "peak_infected": df.Infected.max(),
            "peak_time": df.Infected.idxmax(),
            "total_influence": df.Total_Influence.max(),
            "removed_count": df.Removed.iloc[-1],
            "decay_time": decay_time
        }

    # 创建比较图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 峰值感染数比较
    peaks = [m['peak_infected'] for m in metrics.values()]
    axes[0, 0].bar(metrics.keys(), peaks, color=['blue', 'green', 'red', 'purple'])
    axes[0, 0].set_title('峰值感染节点数')
    axes[0, 0].set_ylabel('节点数')

    # 平息时间比较
    decay_times = [m['decay_time'] for m in metrics.values()]
    axes[0, 1].bar(metrics.keys(), decay_times, color=['blue', 'green', 'red', 'purple'])
    axes[0, 1].set_title('舆情平息时间')
    axes[0, 1].set_ylabel('时间步')

    # 总影响力比较
    influences = [m['total_influence'] for m in metrics.values()]
    axes[1, 0].bar(metrics.keys(), influences, color=['blue', 'green', 'red', 'purple'])
    axes[1, 0].set_title('负面舆情总影响力峰值')
    axes[1, 0].set_ylabel('影响力')

    # 移除节点数比较
    removed = [m['removed_count'] for m in metrics.values()]
    axes[1, 1].bar(metrics.keys(), removed, color=['blue', 'green', 'red', 'purple'])
    axes[1, 1].set_title('平台移除节点数')
    axes[1, 1].set_ylabel('节点数')

    plt.suptitle('不同干预策略效果比较', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("strategy_comparison.png")  # 保存图像
    plt.show()

    return metrics


# 执行策略比较
strategy_metrics = compare_strategies(results)