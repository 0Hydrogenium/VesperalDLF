import torch.nn.functional
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from extra.llm.module.dataset.GPTDataset import GPTDatasetV1


class LLMUtils:

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 768,  # Embedding dim
        "n_heads": 12,  # Number of attn heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # QKV bias
    }

    @classmethod
    def set_config(cls, cfg):
        cls.GPT_CONFIG_124M = cfg

    @classmethod
    def load_weights_into_gpt(cls, gpt_model, params):
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
    def plot_losses(cls, num_epochs, tokens_seen, train_losses, val_losses):
        epochs_seen = torch.linspace(0, num_epochs, len(train_losses))

        fig, ax1 = plt.subplots(figsize=(5, 3))
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha=0)
        ax2.set_xlabel("Tokens seen")
        fig.tight_layout()
        plt.show()

    @classmethod
    def evaluate_model(cls, model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = cls.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = cls.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss

    @classmethod
    def train_model_simple(cls, model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1
        for epoch in range(num_epochs):
            model.train()
            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = cls.calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = cls.evaluate_model(model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch + 1} (Step {global_step:06d}): "f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        cls.generate_and_print_sample(model, tokenizer, device, start_context)
        return train_losses, val_losses, track_tokens_seen

    @classmethod
    def generate_and_print_sample(cls, model, tokenizer, device, start_context):
        model.eval()
        encoded = tokenizer.text_to_token_ids(start_context).to(device)
        with torch.no_grad():
            token_ids = cls.generate(model, encoded, max_new_tokens=50, context_size=cls.GPT_CONFIG_124M["context_length"], temperature=1.4, top_k=25)
            decoded_text = tokenizer.token_ids_to_text(token_ids)
            print(decoded_text.replace("\n", " "))
        model.train()

    @classmethod
    def generate(cls, model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None, eos_id=None):
        # idx: current context index array, (batch, n_tokens)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)

            logits = logits[:, -1, :]  # only focus on the last timestep
            if top_k is not None:
                # top-k sampling strategy
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                # threshold logits by replacing any value below min_val with -inf
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            if temperature > 0:
                # temperature scaling strategy
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                # a function used to sample from a multinomial distribution, randomly drawing sample indices according to a given prob distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # greedy sampling strategy
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if idx_next == eos_id:
                # if encounter end token, generation stops early
                break

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @classmethod
    def calc_loss_loader(cls, data_loader, model, device, num_batches=None):
        total_loss = 0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i <= num_batches:
                loss = cls.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    @classmethod
    def calc_loss_batch(cls, input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    @classmethod
    def create_dataloader_v1(cls, text, tokenizer, batch_size, max_length, stride, shuffle, drop_last=True, num_workers=0):
        dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        return dataloader

    @classmethod
    def compute_model_param_num_and_size(cls, model):
        total_params = sum(p.numel() for p in model.parameters())
        # 假设每个参数为float32类型，占用4个字节
        total_size_bytes = total_params * 4
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        return total_params, total_size_gb
