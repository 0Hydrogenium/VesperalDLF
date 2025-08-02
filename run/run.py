import json

from trainer.CAREWTTrainer import CAREWTTrainer
from trainer.DLClassificationTrainer import DLClassificationTrainer
from trainer.Trainer import Trainer
from trainer.ContrastiveBRAETrainer import ContrastiveBRAETrainer
from trainer.WeibullTwinDAETrainer import WeibullTwinDAETrainer
from utils.GeneralTool import GeneralTool


"""
    1. 分类指标引入数据集不平衡比例
    2. GPT模型中的context_length
    3. 数据分布可视化+时序可视化
"""


if __name__ == '__main__':
    GeneralTool.set_global_seed(6)

    cfg_name = "CAREWT_IsolationForest"
    # cfg_name = "CAREWT_trivial"
    # cfg_name = "WeibullTwinGPT"
    cfg = GeneralTool.load_cfg(cfg_name)

    trainer = CAREWTTrainer(cfg, cfg_name)
    trainer.start()
