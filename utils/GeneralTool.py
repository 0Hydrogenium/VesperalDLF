import os.path
import random
import numpy as np
import torch
import json


class GeneralTool:

    seed = 42  # 全局随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def load_cfg(cls, cfg_name):
        cfg_path = f"{cls.root_path}/config/{cfg_name}.json"
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        return cfg

    @classmethod
    def set_global_seed(cls, seed):
        cls.seed = seed
        # 为所有随机数设置全局随机种子
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)

