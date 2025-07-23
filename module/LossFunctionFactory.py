import os
import importlib

from utils.GeneralTool import GeneralTool


def init_loss_function_factory():
    # 动态扫描并加载所有损失函数类
    folder_path = f"{GeneralTool.root_path}/module/loss_function"
    loss_function_mapping = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".py") and not file_name.startswith("__"):
            class_name = file_name.replace(".py", "")
            try:
                # 动态导入模块
                module = importlib.import_module(f'module.loss_function.{class_name}')
                # 获取同名的类对象
                loss_function_class = getattr(module, class_name)
                loss_function_mapping[class_name] = loss_function_class

            except Exception as e:
                print(e)
                continue

    return loss_function_mapping


class LossFunctionFactory:

    loss_function_mapping = init_loss_function_factory()

    @classmethod
    def get(cls, cfg, loss_function_name):
        loss_function = cls.loss_function_mapping.get(loss_function_name, None)
        return loss_function(cfg)
    