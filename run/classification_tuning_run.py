import json

from extra.llm.trainer.ClassificationTuningTrainer import ClassificationTuningTrainer
from utils.GeneralTool import GeneralTool


if __name__ == '__main__':
    GeneralTool.set_global_seed(42)

    cfg_name = "ClassificationTuning"
    cfg = GeneralTool.load_cfg(cfg_name)

    trainer = ClassificationTuningTrainer(cfg, cfg_name)
    trainer.start()