import json

from trainer.DLClassificationTrainer import DLClassificationTrainer
from trainer.Trainer import Trainer
from trainer.ContrastiveBRAETrainer import ContrastiveBRAETrainer
from trainer.WeibullTwinDAETrainer import WeibullTwinDAETrainer
from utils.GeneralTool import GeneralTool

if __name__ == '__main__':
    GeneralTool.set_global_seed(42)

    cfg_name = "WeibullTwinDAE"
    cfg_path = f"./config/{cfg_name}.json"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    trainer = WeibullTwinDAETrainer(cfg, cfg_name)
    trainer.start()
