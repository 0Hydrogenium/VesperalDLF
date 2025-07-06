import json

from trainer.LSTMTrainer import LSTMTrainer
from trainer.Trainer import Trainer


if __name__ == '__main__':

    cfg_name = "ContrastiveBRAE"
    cfg_path = f"./config/{cfg_name}.json"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    trainer = LSTMTrainer(cfg, cfg_name)
    trainer.start()
