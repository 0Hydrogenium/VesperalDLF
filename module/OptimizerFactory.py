import os
import importlib


def init_optimizer_factory():
    # 动态扫描并加载所有优化器类
    folder_path = "./module/optimizer"
    optimizer_mapping = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".py") and not file_name.startswith("__"):
            class_name = file_name.replace(".py", "")
            try:
                # 动态导入模块
                module = importlib.import_module(f'module.optimizer.{class_name}')
                # 获取同名的类对象
                optimizer_class = getattr(module, class_name)
                optimizer_mapping[class_name] = optimizer_class

            except Exception as e:
                print(e)
                continue

    return optimizer_mapping


class OptimizerFactory:

    optimizer_mapping = init_optimizer_factory()

    @classmethod
    def get(cls, cfg, optimizer_name, model):
        optimizer = cls.optimizer_mapping.get(optimizer_name, None)
        return optimizer(cfg, model)
