import json

from extra.llm.trainer.PretrainTrainer import PretrainTrainer
from utils.GeneralTool import GeneralTool


if __name__ == '__main__':
    GeneralTool.set_global_seed(42)

    cfg_name = "PretrainGPT"
    cfg = GeneralTool.load_cfg(cfg_name)

    trainer = PretrainTrainer(cfg, cfg_name)
    trainer.start()
