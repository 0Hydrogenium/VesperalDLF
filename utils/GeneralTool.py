import random
import numpy as np
import torch


class GeneralTool:

    seed = 42  # 全局随机种子

    @classmethod
    def set_global_seed(cls, seed):
        cls.seed = seed
        # 为所有随机数设置全局随机种子
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
