import pandas as pd
import torch
import optuna
import json
from collections import Counter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score

from module.LossFunctionFactory import LossFunctionFactory
from module.ModelFactory import ModelFactory
from module.OptimizerFactory import OptimizerFactory
from trainer.Trainer import Trainer
from utils.metric_tracker.BestMetricsTracker import BestMetricsTracker
from utils.metric_tracker.ClassificationMetricsTracker import ClassificationMetricsTracker
from utils.metric_tracker.RegressionMetricsTracker import RegressionMetricsTracker
from utils.visualizer.DataMappingVisualizer import DataMappingVisualizer


class ContrastiveBRAETrainer(Trainer):
    def __init__(self, cfg, cfg_name):
        super().__init__(cfg, cfg_name)
        self.visualizer = DataMappingVisualizer()

    def run(self, trial: optuna.trial = None, to_visualize=False):

        # 加载动态参数值
        if trial is not None:
            for param_name, param_value in self.dynamic_params_cfg.items():
                if param_value["type"] == "int":
                    self.cfg[param_name] = trial.suggest_int(param_name, param_value["start"], param_value["end"])
                elif param_value["type"] == "float":
                    self.cfg[param_name] = trial.suggest_float(param_name, param_value["start"], param_value["end"])

        data_info = {}
        # 导入数据表
        df = self.import_data(self.cfg["data_path"])

        # 由于后续使用多头注意力机制，需要将特征数量凑成偶数
        df.insert(df.shape[1], "blank", [0 for _ in range(len(df))])

        print(f"imported data (length: {df.shape[0]}, features: {df.shape[1]})")
        data_info["imported_data_length"] = df.shape[0]
        data_info["imported_data_features"] = df.shape[1]
        # 处理缺失值
        df = self.process_null_value(df)
        print(f"processed null data (length: {df.shape[0]}, features: {df.shape[1]})")
        data_info["processed_null_data_length"] = df.shape[0]
        data_info["processed_null_data_features"] = df.shape[1]
        # 处理目标值
        raw_y_distribution = dict(Counter(df[self.cfg["target_name"]]).most_common())
        print(f"raw y distribution: {raw_y_distribution}")
        data_info["raw_y_distribution"] = raw_y_distribution
        df = self.process_target_value(df)
        print(f"processed target data (length: {df.shape[0]}, features: {df.shape[1]})")
        data_info["processed_target_data_length"] = df.shape[0]
        data_info["processed_target_data_features"] = df.shape[1]
        # 数据标准化
        df = self.normalize_value(df)
        print(f"normalized data (length: {df.shape[0]}, features: {df.shape[1]})")
        data_info["normalized_data_length"] = df.shape[0]
        data_info["normalized_data_features"] = df.shape[1]

        # 分割x和y
        data_x, data_y = self.split_into_x_with_y(df)
        y_distribution = dict(Counter(data_y.tolist()).most_common())
        print(f"y distribution: {y_distribution}")
        data_info["y_distribution"] = y_distribution
        if self.cfg["target_mapping"] == "dynamic_weibull_mapping":
            weibull_data_y, weibull_fault_indices, weibull_baselines, weibull_etas = self.dynamic_weibull_cdf_mapping(
                data_y,
                k=self.cfg["target_mapping_k"],
                cumulative_prob=1,
                degradation_factor=1,
                initial_baseline=0,
                baseline_growth=0
            )
            # 去除最后一个故障之后的正常数据
            data_x = data_x[: len(weibull_data_y), :]
            data_y = data_y[: len(weibull_data_y)]
            if to_visualize:
                self.visualizer.line_chart_mapping_y_distribution_discrete_with_continuous(data_y, weibull_data_y, self.cfg["target_value"], figsize=(14, 6), alpha=0.3, s=4, save_path=f"{self.result_save_path}/y_distribution_plot_trial_{self.trial_idx}.svg")
                self.visualizer.line_chart_mapping_y_distribution_param_trend(weibull_baselines, weibull_etas, self.cfg["target_value"], figsize=(8, 6), alpha=0.3, s=4, save_path1=f"{self.result_save_path}/baseline_param_y_distribution_plot_trial_{self.trial_idx}.svg", save_path2=f"{self.result_save_path}/etas_param_y_distribution_plot_trial_{self.trial_idx}.svg")

            # 分割训练集和测试集
            train_data_x, train_data_y, test_data_x, test_data_y = self.split_into_train_with_test_data(data_x, data_y)
            _, mapped_train_data_y, _, mapped_test_data_y = self.split_into_train_with_test_data(data_x, weibull_data_y)
            # 转换为时间序列数据
            x_train, y_train = self.convert_from_data_to_time_series(train_data_x, train_data_y)
            x_test, y_test = self.convert_from_data_to_time_series(test_data_x, test_data_y)
            _, mapped_y_train = self.convert_from_data_to_time_series(train_data_x, mapped_train_data_y)
            _, mapped_y_test = self.convert_from_data_to_time_series(test_data_x, mapped_test_data_y)
            print(f"converted from data to time series (x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, x_test shape: {x_test.shape}, y_test shape: {y_test.shape}, mapped_y_train shape: {mapped_y_train.shape}, mapped_y_test shape: {mapped_y_test.shape})")
            data_info["x_train_shape"], data_info["y_train_shape"], data_info["x_test_shape"], data_info["y_test_shape"], data_info["mapped_y_train_shape"], data_info["mapped_y_test_shape"] = x_train.shape, y_train.shape, x_test.shape, y_test.shape, mapped_y_train.shape, mapped_y_test.shape
            # ndarray格式转换为tensor格式
            x_train_tensor = self.convert_from_ndarray_to_tensor(x_train)
            y_train_tensor = self.convert_from_ndarray_to_tensor(y_train)
            x_test_tensor = self.convert_from_ndarray_to_tensor(x_test)
            y_test_tensor = self.convert_from_ndarray_to_tensor(y_test)
            mapped_y_train_tensor = self.convert_from_ndarray_to_tensor(mapped_y_train)
            mapped_y_test_tensor = self.convert_from_ndarray_to_tensor(mapped_y_test)
            # 转换为数据加载器
            train_loader, test_loader = self.get_data_loader(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)
            mapped_train_loader, mapped_test_loader = self.get_data_loader(x_train_tensor, mapped_y_train_tensor, x_test_tensor, mapped_y_test_tensor)

        else:
            # 分割训练集和测试集
            train_data_x, train_data_y, test_data_x, test_data_y = self.split_into_train_with_test_data(data_x, data_y)
            # 转换为时间序列数据
            x_train, y_train = self.convert_from_data_to_time_series(train_data_x, train_data_y)
            x_test, y_test = self.convert_from_data_to_time_series(test_data_x, test_data_y)
            print(f"converted from data to time series (x_train shape: {x_train.shape}, y_train shape: {y_train.shape}, x_test shape: {x_test.shape}, y_test shape: {y_test.shape})")
            data_info["x_train_shape"], data_info["y_train_shape"], data_info["x_test_shape"], data_info["y_test_shape"] = x_train.shape, y_train.shape, x_test.shape, y_test.shape
            # ndarray格式转换为tensor格式
            x_train_tensor = self.convert_from_ndarray_to_tensor(x_train)
            y_train_tensor = self.convert_from_ndarray_to_tensor(y_train)
            x_test_tensor = self.convert_from_ndarray_to_tensor(x_test)
            y_test_tensor = self.convert_from_ndarray_to_tensor(y_test)
            # 转换为数据加载器
            train_loader, test_loader = self.get_data_loader(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)

        p_x_train_tensor, p_y_train_tensor, n_x_train_tensor, n_y_train_tensor = self.split_into_positive_with_negative(x_train_tensor, y_train_tensor)
        p_x_test_tensor, p_y_test_tensor, n_x_test_tensor, n_y_test_tensor = self.split_into_positive_with_negative(x_test_tensor, y_test_tensor)

        p_x_array = torch.concat((p_x_train_tensor, p_x_test_tensor), dim=0).detach().cpu().numpy()

        p_train_loader, p_test_loader = self.get_data_loader(p_x_train_tensor, p_y_train_tensor, p_x_test_tensor, p_y_test_tensor)
        n_train_loader, n_test_loader = self.get_data_loader(n_x_train_tensor, n_y_train_tensor, n_x_test_tensor, n_y_test_tensor)

        # 根据数据获取模型的输入维度和输出维度
        self.cfg["model__input_dim"] = data_x.shape[1]
        self.cfg["model__output_dim"] = 2
        self.cfg["model__threshold_output_dim"] = 1

        # 创建模型
        p_model = ModelFactory.get(self.cfg, self.cfg["positive_model_name"])

        n_model = ModelFactory.get(self.cfg, self.cfg["negative_model_name"]).to(self.device)
        n_loss_function = LossFunctionFactory.get(self.cfg, self.cfg["loss_function_name"])
        n_optimizer = OptimizerFactory.get(self.cfg, self.cfg["optimizer_name"], n_model)

        threshold_model = ModelFactory.get(self.cfg, self.cfg["threshold_model_name"]).to(self.device)
        threshold_loss_function = LossFunctionFactory.get(self.cfg, self.cfg["loss_function_name"])
        threshold_optimizer = OptimizerFactory.get(self.cfg, self.cfg["optimizer_name"], threshold_model)

        # 迭代训练测试模型
        # 创建epochs内最佳指标追踪器
        n_best_metrics_tracker = BestMetricsTracker()
        threshold_best_metrics_tracker = BestMetricsTracker()
        best_metrics_tracker = BestMetricsTracker()
        for epoch in range(1, self.cfg["epochs"] + 1):
            """
                训练测试负样本重构模型
            """
            # 训练负重构模型
            n_train_metrics_tracker, n_model = self.train_negative_model(n_model, n_loss_function, n_optimizer, n_train_loader, epoch)
            print(f"[train] negative metrics: {n_train_metrics_tracker.get_metrics()}\n")
            # 测试负重构模型
            n_test_metrics_tracker = self.test_negative_model(n_model, n_loss_function, n_test_loader, epoch)
            print(f"[test] negative metrics: {n_test_metrics_tracker.get_metrics()}\n")
            # 更新epochs内最佳模型+参数+指标
            n_best_metrics_tracker.add(n_train_metrics_tracker, n_test_metrics_tracker, self.cfg, epoch)

            """
                训练测试正样本重构模型：只在第一个epoch训练模型
            """
            # 训练正重构模型
            if epoch == 1:
                p_baseline_nll, p_train_nll, p_train_metrics, p_model = self.train_positive_model(p_model, p_x_array)
                print(f"[train] positive metrics: {p_train_metrics}\n")

            """
                计算偏差获取特征阈值
            """
            # 训练集计算负样本偏差
            n_data_deviation = self.compute_deviation(p_model, n_model, n_train_loader, decay=0.8)
            # 训练集计算正样本偏差
            p_data_deviation = self.compute_deviation(p_model, n_model, p_train_loader, decay=0.8)
            # 训练集计算特征阈值
            feature_threshold = self.compute_feature_threshold(p_data_deviation, n_data_deviation)

            """
                训练测试阈值模型
            """
            if self.cfg["target_mapping"] == "dynamic_weibull_mapping":
                # 训练集计算偏差
                mapped_train_deviation = self.compute_deviation(p_model, n_model, mapped_train_loader, decay=0.8)
                # 测试集计算偏差
                mapped_test_deviation = self.compute_deviation(p_model, n_model, mapped_test_loader, decay=0.8)

                # # 偏差的对应特征减去特征阈值
                # mapped_train_deviation = mapped_train_deviation - feature_threshold

                # mapped_test_deviation = mapped_test_deviation - feature_threshold

                # 标准化偏差
                mapped_train_deviation = self.normalize_value(mapped_train_deviation)
                mapped_test_deviation = self.normalize_value(mapped_test_deviation)
                # 转换为时间序列偏差
                mapped_train_deviation, _ = self.convert_from_data_to_time_series(mapped_train_deviation)
                mapped_test_deviation, _ = self.convert_from_data_to_time_series(mapped_test_deviation)
                # ndarray格式转换为tensor格式
                mapped_train_deviation_tensor = self.convert_from_ndarray_to_tensor(mapped_train_deviation)
                mapped_test_deviation_tensor = self.convert_from_ndarray_to_tensor(mapped_test_deviation)
                # 转换为数据加载器
                combined_train_dataset = torch.utils.data.TensorDataset(mapped_train_deviation_tensor, y_train_tensor, mapped_y_train_tensor)
                combined_test_dataset = torch.utils.data.TensorDataset(mapped_test_deviation_tensor, y_test_tensor, mapped_y_test_tensor)
                combined_train_loader = torch.utils.data.DataLoader(combined_train_dataset, self.cfg["batch_size"], True)
                combined_test_loader = torch.utils.data.DataLoader(combined_test_dataset, self.cfg["batch_size"], False)


                # 训练阈值模型
                threshold_train_metrics_tracker, threshold_model, combined_y_train_array, combined_x_train_pred_array = self.train_threshold_model(threshold_model, threshold_loss_function, threshold_optimizer, combined_train_loader, epoch)
                print(f"[train] threshold metrics: {threshold_train_metrics_tracker.get_metrics()}\n")
                # 测试阈值模型
                threshold_test_metrics_tracker, combined_y_test_array, combined_x_test_pred_array = self.test_threshold_model(threshold_model, threshold_loss_function, combined_test_loader, epoch)
                print(f"[test] threshold metrics: {threshold_test_metrics_tracker.get_metrics()}\n")
                # 更新epochs内最佳模型+参数+指标
                threshold_best_metrics_tracker.add(threshold_train_metrics_tracker, threshold_test_metrics_tracker, self.cfg, epoch)

                """
                    计算ROC曲线相关参数指标:
                
                    遍历所有唯一的预测概率值作为阈值
                    剔除次优阈值
                    若相邻阈值的(FPR,TPR)相同或线性相关，则合并中间阈值
                    添加额外两个固定端点
                """
                train_fpr, train_tpr, train_thresholds, train_roc_auc = self.compute_roc_auc(combined_y_train_array, combined_x_train_pred_array)
                test_fpr, test_tpr, test_thresholds, test_roc_auc = self.compute_roc_auc(combined_y_test_array, combined_x_test_pred_array)
                # 计算不同阈值下的分类指标
                total_train_metrics_tracker, train_threshold_accuracy, train_threshold_precision, train_threshold_recall, train_threshold_f1 = self.compute_classification_metrics_on_thresholds(combined_y_train_array, combined_x_train_pred_array, train_thresholds)
                total_test_metrics_tracker, test_threshold_accuracy, test_threshold_precision, test_threshold_recall, test_threshold_f1 = self.compute_classification_metrics_on_thresholds(combined_y_test_array, combined_x_test_pred_array, test_thresholds)
                # 计算PR曲线的AP分数
                train_pr_ap = self.compute_pr_ap(train_threshold_precision, train_threshold_recall)
                test_pr_ap = self.compute_pr_ap(test_threshold_precision, test_threshold_recall)

                total_train_metrics_tracker.add_new_metric("roc_auc", train_roc_auc)
                total_test_metrics_tracker.add_new_metric("roc_auc", test_roc_auc)
                total_train_metrics_tracker.add_new_metric("pr_ap", train_pr_ap)
                total_test_metrics_tracker.add_new_metric("pr_ap", test_pr_ap)

                # 更新epochs内最佳模型+参数+指标
                best_metrics_tracker.add(total_train_metrics_tracker, total_test_metrics_tracker, self.cfg, epoch)

                if to_visualize:
                    # 绘制ROC曲线
                    self.visualizer.roc_curve_on_thresholds(train_fpr, train_tpr, train_roc_auc, figsize=(8, 6), save_path=f"{self.result_save_path}/train_roc_curve_on_thresholds_epoch_{epoch}.svg")
                    self.visualizer.roc_curve_on_thresholds(test_fpr, test_tpr, test_roc_auc, figsize=(8, 6), save_path=f"{self.result_save_path}/test_roc_curve_on_thresholds_epoch_{epoch}.svg")
                    # 绘制PR曲线
                    self.visualizer.pr_curve_on_thresholds(train_threshold_recall, train_threshold_precision, train_pr_ap, sum(train_data_y) / len(train_data_y), figsize=(8, 6), save_path=f"{self.result_save_path}/train_pr_curve_on_thresholds_epoch_{epoch}.svg")
                    self.visualizer.pr_curve_on_thresholds(test_threshold_recall, test_threshold_precision, test_pr_ap, sum(test_data_y) / len(test_data_y), figsize=(8, 6), save_path=f"{self.result_save_path}/test_pr_curve_on_thresholds_epoch_{epoch}.svg")

        # 获取epochs内最佳模型+参数+指标
        best_metrics, best_cfg, best_epoch = best_metrics_tracker.get_best(self.cfg["metrics_filter"], self.cfg["optimize_metric"], self.cfg["optimize_direction"])
        print(f"[best] epoch of metrics: {best_epoch}")
        print(f"[best] metrics: {best_metrics}\n")
        # 保存所有epoch数据
        n_best_metrics_tracker.save_data(f"{self.result_save_path}/negative_model_trial_{self.trial_idx}.csv")
        threshold_best_metrics_tracker.save_data(f"{self.result_save_path}/threshold_model_trial_{self.trial_idx}.csv")
        best_metrics_tracker.save_data(f"{self.result_save_path}/trial_{self.trial_idx}.csv")

        # 保存数据信息
        with open(f"{self.result_save_path}/data_info_trial_{self.trial_idx}.json", 'w', encoding='utf-8') as f:
            json.dump(data_info, f, ensure_ascii=False, indent=4)

        if isinstance(self.trial_idx, int):
            # trial_idx代表超参数优化次数
            self.trial_idx += 1

        # 无可行指标组合处理
        if best_epoch is None:
            return tuple(0 for _ in self.cfg["optimize_metric"])

        # 绘制模型训练测试曲线
        if to_visualize:
            # 负重构模型
            n_train_loss_mean = np.array([sum(loss) / len(loss) for loss in [
                tracker.metrics["loss"] for tracker in n_best_metrics_tracker.train_metrics_tracker_list
            ]])
            n_train_loss_std = np.array([[np.std(np.array(loss)) for loss in [
                tracker.metrics["loss"] for tracker in n_best_metrics_tracker.train_metrics_tracker_list
            ]]]).reshape(-1)
            n_test_loss_mean = np.array([sum(loss) / len(loss) for loss in [
                tracker.metrics["loss"] for tracker in n_best_metrics_tracker.test_metrics_tracker_list
            ]])
            n_test_loss_std = np.array([[np.std(np.array(loss)) for loss in [
                tracker.metrics["loss"] for tracker in n_best_metrics_tracker.test_metrics_tracker_list
            ]]]).reshape(-1)

            self.visualizer.loss_plot_train_test_curve_with_std(
                n_train_loss_mean,
                n_train_loss_std,
                n_test_loss_mean,
                n_test_loss_std,
                scale=0.5,
                alpha=0.2,
                figsize=(8, 6),
                save_path=f"{self.result_save_path}/negative_model_loss_plot.svg"
            )

            # 阈值模型
            threshold_train_loss_mean = np.array([sum(loss) / len(loss) for loss in [
                tracker.metrics["loss"] for tracker in threshold_best_metrics_tracker.train_metrics_tracker_list
            ]])
            threshold_train_loss_std = np.array([[np.std(np.array(loss)) for loss in [
                tracker.metrics["loss"] for tracker in threshold_best_metrics_tracker.train_metrics_tracker_list
            ]]]).reshape(-1)
            threshold_test_loss_mean = np.array([sum(loss) / len(loss) for loss in [
                tracker.metrics["loss"] for tracker in threshold_best_metrics_tracker.test_metrics_tracker_list
            ]])
            threshold_test_loss_std = np.array([[np.std(np.array(loss)) for loss in [
                tracker.metrics["loss"] for tracker in threshold_best_metrics_tracker.test_metrics_tracker_list
            ]]]).reshape(-1)

            self.visualizer.loss_plot_train_test_curve_with_std(
                threshold_train_loss_mean,
                threshold_train_loss_std,
                threshold_test_loss_mean,
                threshold_test_loss_std,
                scale=0.5,
                alpha=0.2,
                figsize=(8, 6),
                save_path=f"{self.result_save_path}/threshold_model_loss_plot.svg"
            )

        return tuple(best_metrics[metric] for metric in self.cfg["optimize_metric"])

    def compute_classification_metrics_on_thresholds(self, data_y, data_y_proba, thresholds):
        accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
        metrics_tracker = ClassificationMetricsTracker()
        for idx, threshold in enumerate(thresholds):
            # 应用当前阈值进行二分类
            pred = (data_y_proba >= threshold).astype(int)
            # 分类指标器中的无效值不考虑，但是函数返回的分类指标的数量需要相同，因此各自分别计算
            # 分类指标器计算的是代表性指标值
            # 而独立计算的分类指标为返回数组
            metrics_tracker.update(real_array=data_y, pred_array=pred)

            accuracy = accuracy_score(y_true=data_y, y_pred=pred)
            precision = precision_score(y_true=data_y, y_pred=pred, zero_division=0)
            recall = recall_score(y_true=data_y, y_pred=pred, zero_division=0)
            f1 = f1_score(y_true=data_y, y_pred=pred, zero_division=0)

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        threshold_accuracy = np.array(accuracy_list)
        threshold_precision = np.array(precision_list)
        threshold_recall = np.array(recall_list)
        threshold_f1 = np.array(f1_list)

        return metrics_tracker, threshold_accuracy, threshold_precision, threshold_recall, threshold_f1

    def compute_roc_auc(self, data_y, data_y_proba):
        fpr, tpr, thresholds_roc = roc_curve(data_y, data_y_proba)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds_roc, roc_auc

    def compute_pr_ap(self, precision, recalls):
        # 梯度积分法计算AP分数
        ap = auc(recalls, precision)
        return ap

    def compute_feature_threshold(self, p_data_deviation, n_data_deviation):
        feature_threshold_list = []
        for feature_idx in range(n_data_deviation.shape[1]):
            p_data_median = np.median(p_data_deviation[:, feature_idx].reshape(-1))
            n_data_median = np.median(n_data_deviation[:, feature_idx].reshape(-1))
            feature_threshold = (p_data_median + n_data_median) / 2
            feature_threshold_list.append(feature_threshold)
        return np.array(feature_threshold_list)

    def compute_deviation(self, p_model, n_model, loader, decay):
        deviation_list = []
        # 计算偏差时不更新模型参数
        with torch.no_grad():
            for idx, tensors in enumerate(loader):
                x_array, y_array = tensors
                x_array, y_array = x_array.to(self.device), y_array.to(self.device)
                n_model_x_array_pred, _ = n_model(x_array)

                n_model_x_array_pred = n_model_x_array_pred.detach().cpu().numpy()
                x_array = x_array.detach().cpu().numpy()

                p_model_x_array_pred = p_model.test(x_array.reshape(x_array.shape[0], -1))
                p_model_x_array_pred = p_model_x_array_pred.reshape(-1, x_array.shape[1], x_array.shape[2])

                # 偏差一定为正数
                n_model_deviation = np.abs(n_model_x_array_pred - x_array)
                p_model_deviation = np.abs(p_model_x_array_pred - x_array)
                # 偏差是相对于负样本的
                deviation = n_model_deviation - self.cfg["model__threshold_theta"] * p_model_deviation
                deviation_list.append(deviation)

        deviation_array = np.concatenate(deviation_list, axis=0)

        # 在时间窗口维度使用指数衰减法，考虑序列数据的时间价值递减
        weight_array = decay ** np.arange(deviation_array.shape[1])
        # 采用广播机制
        deviation_array = deviation_array * weight_array.reshape(1, deviation_array.shape[1], 1)
        deviation_array = np.sum(deviation_array, axis=1)
        return deviation_array

    def train_threshold_model(self, model, loss_function, optimizer, train_loader, epoch):
        model.train()
        train_metrics_tracker = RegressionMetricsTracker()
        loss_list = []
        train_bar = tqdm(train_loader)
        y_train_list = []
        x_train_pred_list = []
        for idx, train_tensors in enumerate(train_bar):
            x_train, y_train, mapped_y_train = train_tensors
            x_train, mapped_y_train = x_train.to(self.device), mapped_y_train.to(self.device)
            optimizer.zero_grad()
            x_train_pred = model(x_train)
            loss = loss_function(x_train_pred, mapped_y_train.reshape(-1, 1))
            loss.backward()
            optimizer.step()

            mapped_y_train = mapped_y_train.detach().cpu().numpy()
            x_train_pred = x_train_pred.detach().cpu().numpy()
            y_train = y_train.detach().cpu().numpy()

            train_metrics_tracker.update(real_array=mapped_y_train.reshape(mapped_y_train.shape[0], -1), pred_array=x_train_pred.reshape(x_train_pred.shape[0], -1))
            loss_list.append(loss.item())

            y_train_list.append(y_train)
            x_train_pred_list.append(x_train_pred)

            train_bar.desc = "threshold train epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        y_train_array = np.concatenate(y_train_list, axis=0).reshape(-1, 1)
        x_train_pred_array = np.concatenate(x_train_pred_list, axis=0)

        train_metrics_tracker.add_new_metrics("loss", loss_list)
        return train_metrics_tracker, model, y_train_array, x_train_pred_array

    def test_threshold_model(self, model, loss_function, test_loader, epoch):
        model.eval()
        test_metrics_tracker = RegressionMetricsTracker()
        loss_list = []
        test_bar = tqdm(test_loader)
        y_test_list = []
        x_test_pred_list = []
        with torch.no_grad():
            for idx, test_tensors in enumerate(test_bar):
                x_test, y_test, mapped_y_test = test_tensors
                x_test, mapped_y_test = x_test.to(self.device), mapped_y_test.to(self.device)
                x_test_pred = model(x_test)
                loss = loss_function(x_test_pred, mapped_y_test.reshape(-1, 1))

                mapped_y_test = mapped_y_test.detach().cpu().numpy()
                x_test_pred = x_test_pred.detach().cpu().numpy()
                y_test = y_test.detach().cpu().numpy()

                test_metrics_tracker.update(real_array=mapped_y_test.reshape(mapped_y_test.shape[0], -1), pred_array=x_test_pred.reshape(x_test_pred.shape[0], -1))
                loss_list.append(loss.item())

                y_test_list.append(y_test)
                x_test_pred_list.append(x_test_pred)

                test_bar.desc = "threshold test epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)

        y_test_array = np.concatenate(y_test_list, axis=0).reshape(-1, 1)
        x_test_pred_array = np.concatenate(x_test_pred_list, axis=0)

        test_metrics_tracker.add_new_metrics("loss", loss_list)
        return test_metrics_tracker, y_test_array, x_test_pred_array

    def train_positive_model(self, model, x_train_array):
        train_metrics_tracker = RegressionMetricsTracker()

        x_train_array = x_train_array.reshape(x_train_array.shape[0], -1)
        model.experience_params_setting(x_train_array)
        baseline_nll = model.compute_baseline_nll(x_train_array)
        x_train_pred_array = model.train(x_train_array)
        train_nll = model.compute_negative_log_likelihood(x_train_array, x_train_array)

        train_metrics_tracker.update(real_array=x_train_array, pred_array=x_train_pred_array)
        return baseline_nll, train_nll, train_metrics_tracker.get_metrics(), model

    def train_negative_model(self, model, loss_function, optimizer, train_loader, epoch):
        model.train()
        train_metrics_tracker = RegressionMetricsTracker()
        loss_list = []
        train_bar = tqdm(train_loader)
        for idx, train_tensors in enumerate(train_bar):
            x_train, y_train = train_tensors
            x_train, y_train = x_train.to(self.device), y_train.to(self.device)
            optimizer.zero_grad()
            x_train_pred, kl = model(x_train)
            kl = 0.01 * min(epoch / self.cfg["epochs"], 1)
            loss = loss_function(x_train_pred, x_train) + kl
            loss.backward()
            optimizer.step()

            y_train = y_train.detach().cpu().numpy()
            x_train = x_train.detach().cpu().numpy()
            x_train_pred = x_train_pred.detach().cpu().numpy()

            train_metrics_tracker.update(real_array=x_train.reshape(x_train.shape[0], -1), pred_array=x_train_pred.reshape(x_train_pred.shape[0], -1))
            loss_list.append(loss.item())
            train_bar.desc = "train epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)
        train_metrics_tracker.add_new_metrics("loss", loss_list)
        return train_metrics_tracker, model

    def test_negative_model(self, model, loss_function, test_loader, epoch):
        model.eval()
        test_metrics_tracker = RegressionMetricsTracker()
        loss_list = []
        test_bar = tqdm(test_loader)
        with torch.no_grad():
            for idx, test_tensors in enumerate(test_bar):
                x_test, y_test = test_tensors
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                x_test_pred, kl = model(x_test)
                loss = loss_function(x_test_pred, x_test) + kl

                y_test = y_test.detach().cpu().numpy()
                x_test = x_test.detach().cpu().numpy()
                x_test_pred = x_test_pred.detach().cpu().numpy()

                test_metrics_tracker.update(real_array=x_test.reshape(x_test.shape[0], -1), pred_array=x_test_pred.reshape(x_test_pred.shape[0], -1))
                loss_list.append(loss.item())
                test_bar.desc = "test epoch[{}/{}] (init:{:4f}) loss:{:.4f}".format(epoch, self.cfg["epochs"], loss_list[0], loss)
        test_metrics_tracker.add_new_metrics("loss", loss_list)
        return test_metrics_tracker

    def split_into_positive_with_negative(self, x_tensor, y_tensor):
        # 创建布尔掩码
        positive_mask = (y_tensor == 1)
        negative_mask = ~positive_mask

        p_x_tensor = x_tensor[positive_mask]
        p_y_tensor = y_tensor[positive_mask]
        n_x_tensor = x_tensor[negative_mask]
        n_y_tensor = y_tensor[negative_mask]

        return p_x_tensor, p_y_tensor, n_x_tensor, n_y_tensor

    def dynamic_weibull_cdf_mapping(self, data, k=2, cumulative_prob=0.99, degradation_factor=0.85, initial_baseline=0.0, baseline_growth=0.05):
        """

        :param data: 0/1序列
        :param k: Weibull分布形状参数 (k>1)
        :param cumulative_prob: 故障点目标累积概率
        :param degradation_factor: 设备状态退化因子 (0.8-0.95)
        :param initial_baseline: 初始基线故障概率 (0-1)
        :param baseline_growth: 每次维修后基线故障的增幅 (0-0.2)
        :return: 故障概率序列 (0-1), 故障点位置, 各段基线故障概率, 各段特征寿命
        """

        fault_indices = np.where(data == 1)[0]

        # 处理无故障情况
        if len(fault_indices) == 0:
            return np.zeros_like(data), fault_indices, [], []

        # 截断到最后一个故障点
        last_fault_idx = fault_indices[-1]
        data = data[:last_fault_idx + 1]
        fault_indices = fault_indices[fault_indices <= last_fault_idx]

        # 初始化结果数组
        mapped_data = np.zeros(len(data), dtype=float)
        baselines = []  # 存储各段起始基线故障概率
        etas = []  # 存储各段特征寿命

        # 计算基数
        log_base = - np.log(1 - cumulative_prob)
        # 当前基线概率（随时间增加）
        current_baseline = initial_baseline
        # 处理所有故障段
        for i, fault_idx in enumerate(fault_indices):
            # 确定当前段起点
            if i == 0:
                start_idx = 0
            else:
                start_idx = fault_indices[i - 1] + 1

            # 记录当前段基线值
            baselines.append(current_baseline)
            # 计算当前段运行时间
            operation_time = fault_idx - start_idx

            # 计算当前段特征寿命
            if operation_time == 0:
                # 特征寿命极短 (快速失效)
                eta = 0.1
            else:
                # 计算基础特征寿命
                base_eta = operation_time / (log_base ** (1 / k))
                # 应用退化效应: 随着故障次数增加，特征寿命缩短
                eta = base_eta * (degradation_factor ** i)
                # 考虑基线的影响 (基线越高，特征寿命越短)
                if current_baseline > 0.3:
                    eta *= (1 - current_baseline)

            etas.append(eta)

            # 计算当前段累积分布函数CDF (从当前基线值开始增长)
            t_segment = np.arange(operation_time + 1)
            # 计算Weibull分布的值 (从0到1)
            weibull_cdf = 1 - np.exp(- (t_segment / eta) ** k)
            # 从基线值开始增长到目标累积概率
            adjusted_cdf = current_baseline + (cumulative_prob - current_baseline) * weibull_cdf
            # 更新结果 (仅覆盖当前段)
            mapped_data[start_idx: start_idx + len(adjusted_cdf)] = adjusted_cdf
            # 更新基线概率: 每次维修后基线增加 (设备整体状态恶化)
            # current_baseline += baseline_growth
            current_baseline *= (1 + baseline_growth)
            # current_baseline = min(current_baseline, 0.5)  # 上限控制

        # 确保段内单调递增 (但允许段间下降)
        for i, fault_idx in enumerate(fault_indices):
            if i == 0:
                start_idx = 0
            else:
                start_idx = fault_indices[i - 1] + 1

            segment = mapped_data[start_idx: fault_idx + 1]

            # 仅保证段内单调递增
            mapped_data[start_idx: fault_idx + 1] = np.maximum.accumulate(segment)

        return mapped_data, fault_indices, baselines, etas
