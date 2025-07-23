import json

from extra.llm.trainer.PretrainTrainer import PretrainTrainer
from utils.GeneralTool import GeneralTool


if __name__ == '__main__':
    GeneralTool.set_global_seed(42)

    cfg_name = "GPT2LLM"
    cfg_path = f"./extra/llm/config/{cfg_name}.json"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    trainer = PretrainTrainer(cfg, cfg_name)
    trainer.start()
