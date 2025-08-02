import os
import importlib
import torch

from utils.GeneralTool import GeneralTool


def init_model_factory():
    # 动态扫描并加载所有模型类
    folder_path = f"{GeneralTool.root_path}/module/model"
    model_mapping = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".py") and not file_name.startswith("__"):
            class_name = file_name.replace(".py", "")
            try:
                # 动态导入模块
                module = importlib.import_module(f'module.model.{class_name}')
                # 获取同名的类对象
                model_class = getattr(module, class_name)
                model_mapping[class_name] = model_class

            except Exception as e:
                print(e)
                continue

    return model_mapping


class ModelFactory:

    model_mapping = init_model_factory()

    @classmethod
    def get(cls, cfg, model_name, model_save_path=None):
        model = cls.model_mapping.get(model_name, None)
        if model is not None and model_save_path is not None:
            # 加载本地模型权重
            model.load_state_dict(torch.load(model_save_path))
        return model(cfg)
