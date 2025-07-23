from extra.llm.gpt2.gpt_download import download_and_load_gpt2
from extra.llm.module.model.GPTModel import GPTModel
from extra.llm.LLMUtils import LLMUtils


class PretrainGPT2Utils:
    @classmethod
    def build(cls):
        gpt_settings, gpt_params = download_and_load_gpt2(model_size="124M", models_dir="gpt2_state_dict")
        print("Settings:", gpt_settings)  # model structure
        print("Parameter dictionary keys:", gpt_params.keys())  # layer params

        model_configs = {
            "gpt2_state_dict-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "qkv_bias": True,
                                             "context_length": 1024},
            "gpt2_state_dict-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "qkv_bias": True,
                                              "context_length": 1024},
            "gpt2_state_dict-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "qkv_bias": True,
                                             "context_length": 1024},
            "gpt2_state_dict-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "qkv_bias": True,
                                           "context_length": 1024},
        }

        model_name = "gpt2_state_dict-small (124M)"
        LLMUtils.GPT_CONFIG_124M.update(model_configs[model_name])
        gpt_model = GPTModel(LLMUtils.GPT_CONFIG_124M)
        LLMUtils.load_weights_into_gpt(gpt_model, gpt_params)
        return gpt_model, LLMUtils.GPT_CONFIG_124M


