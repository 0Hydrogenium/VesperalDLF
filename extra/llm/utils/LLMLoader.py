import json
import numpy as np
import tensorflow as tf
import os
import torch

from extra.llm.module.model.GPTModel import GPTModel
from utils.GeneralTool import GeneralTool


class LLMLoader:
    @classmethod
    def load(cls, model_name, cfg):
        if model_name in ["GPT2-124M", "GPT2-355M"]:
            gpt_settings, gpt_params = cls._load_gpt2(model_name)
            print("Settings:", gpt_settings)
            print("Parameter dictionary keys:", gpt_params.keys())

            model_configs = {
                "GPT2-124M": {
                    "emb_dim": 768,
                    "n_layers": 12,
                    "n_heads": 12,
                    "qkv_bias": True,
                    "context_length": 1024
                },
                "GPT2-355M": {
                    "emb_dim": 1024,
                    "n_layers": 24,
                    "n_heads": 16,
                    "qkv_bias": True,
                    "context_length": 1024
                },
            }

            cfg.update(model_configs[model_name])
            gpt_model = GPTModel(cfg)
            cls._load_weights_into_gpt(gpt_model, gpt_params)
            return gpt_model, cfg

    @classmethod
    def compute_model_param_num_and_size(cls, model):
        total_params = sum(p.numel() for p in model.parameters())
        # 假设每个参数为float32类型，占用4个字节
        total_size_bytes = total_params * 4
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        return total_params, total_size_gb

    @classmethod
    def _load_weights_into_gpt(cls, gpt_model, params):
        gpt_model.pos_emb.weight = cls.assign_vectors_shape(gpt_model.pos_emb.weight, params["wpe"])  # position embedding
        gpt_model.tok_emb.weight = cls.assign_vectors_shape(gpt_model.tok_emb.weight, params["wte"])  # token embedding
        for block in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split((params["blocks"][block]["attn"]["c_attn"])["w"], 3, axis=-1)
            gpt_model.trf_blocks[block].attn.W_query.weight = cls.assign_vectors_shape(gpt_model.trf_blocks[block].attn.W_query.weight, q_w.T)
            gpt_model.trf_blocks[block].attn.W_key.weight = cls.assign_vectors_shape(gpt_model.trf_blocks[block].attn.W_key.weight, k_w.T)
            gpt_model.trf_blocks[block].attn.W_value.weight = cls.assign_vectors_shape(gpt_model.trf_blocks[block].attn.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split((params["blocks"][block]["attn"]["c_attn"])["b"], 3, axis=-1)
            gpt_model.trf_blocks[block].attn.W_query.bias = cls.assign_vectors_shape(gpt_model.trf_blocks[block].attn.W_query.bias, q_b)
            gpt_model.trf_blocks[block].attn.W_key.bias = cls.assign_vectors_shape(gpt_model.trf_blocks[block].attn.W_key.bias, k_b)
            gpt_model.trf_blocks[block].attn.W_value.bias = cls.assign_vectors_shape(gpt_model.trf_blocks[block].attn.W_value.bias, v_b)

            gpt_model.trf_blocks[block].attn.out_proj.weight = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].attn.out_proj.weight, params["blocks"][block]["attn"]["c_proj"]["w"].T
            )
            gpt_model.trf_blocks[block].attn.out_proj.bias = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].attn.out_proj.bias, params["blocks"][block]["attn"]["c_proj"]["b"]
            )

            gpt_model.trf_blocks[block].ff.layers[0].weight = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].ff.layers[0].weight, params["blocks"][block]["mlp"]["c_fc"]["w"].T
            )
            gpt_model.trf_blocks[block].ff.layers[0].bias = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].ff.layers[0].bias, params["blocks"][block]["mlp"]["c_fc"]["b"]
            )

            gpt_model.trf_blocks[block].ff.layers[2].weight = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].ff.layers[2].weight, params["blocks"][block]["mlp"]["c_proj"]["w"].T
            )
            gpt_model.trf_blocks[block].ff.layers[2].bias = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].ff.layers[2].bias, params["blocks"][block]["mlp"]["c_proj"]["b"]
            )

            gpt_model.trf_blocks[block].norm1.scale = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].norm1.scale, params["blocks"][block]["ln_1"]["g"]
            )
            gpt_model.trf_blocks[block].norm1.shift = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].norm1.shift, params["blocks"][block]["ln_1"]["b"]
            )

            gpt_model.trf_blocks[block].norm2.scale = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].norm2.scale, params["blocks"][block]["ln_2"]["g"]
            )
            gpt_model.trf_blocks[block].norm2.shift = cls.assign_vectors_shape(
                gpt_model.trf_blocks[block].norm2.shift, params["blocks"][block]["ln_2"]["b"]
            )

        gpt_model.final_norm.scale = cls.assign_vectors_shape(gpt_model.final_norm.scale, params["g"])
        gpt_model.final_norm.shift = cls.assign_vectors_shape(gpt_model.final_norm.shift, params["b"])
        gpt_model.out_head.weight = cls.assign_vectors_shape(gpt_model.out_head.weight, params["wte"])  # params sharing

    @classmethod
    def assign_vectors_shape(cls, left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))

    @classmethod
    def _load_gpt2(cls, model_name):
        model_dir = f"{GeneralTool.root_path}/model/{model_name}"
        tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
        settings = json.load(open(f"{model_dir}/hparams.json", "r", encoding="utf-8"))
        params = cls._load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
        return settings, params

    @classmethod
    def _load_gpt2_params_from_tf_ckpt(cls, ckpt_path, settings):
        # Initialize parameters dictionary with empty blocks for each layer
        params = {"blocks": [{} for _ in range(settings["n_layer"])]}

        # Iterate over each variable in the checkpoint
        for name, _ in tf.train.list_variables(ckpt_path):
            # Load the variable and remove singleton dimensions
            variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

            # Process the variable name to extract relevant parts
            variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

            # Identify the target dictionary for the variable
            target_dict = params
            if variable_name_parts[0].startswith("h"):
                layer_number = int(variable_name_parts[0][1:])
                target_dict = params["blocks"][layer_number]

            # Recursively access or create nested dictionaries
            for key in variable_name_parts[1:-1]:
                target_dict = target_dict.setdefault(key, {})

            # Assign the variable array to the last key
            last_key = variable_name_parts[-1]
            target_dict[last_key] = variable_array

        return params

