import torch


class TextGenerator:

    @classmethod
    def generate_text(cls, input_text, model, tokenizer, device, max_new_tokens, context_length, temperature, top_k):
        model.eval()
        encoded = tokenizer.text_to_token_ids(input_text).to(device)
        with torch.no_grad():
            token_ids = cls.generate(model, encoded, max_new_tokens=max_new_tokens, context_length=context_length, temperature=temperature, top_k=top_k)
            decoded_text = tokenizer.token_ids_to_text(token_ids)
        return decoded_text

    @classmethod
    def generate(cls, model, idx, max_new_tokens, context_length, temperature=1.0, top_k=None, eos_id=None):
        # idx: current context index array, (batch, n_tokens)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]
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
