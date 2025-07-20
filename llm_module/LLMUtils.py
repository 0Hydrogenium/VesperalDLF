class LLMUtils:
    @classmethod
    def compute_model_param_num_and_size(cls, model):
        total_params = sum(p.numel() for p in model.parameters())
        # 假设每个参数为float32类型，占用4个字节
        total_size_bytes = total_params * 4
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        return total_params, total_size_gb
