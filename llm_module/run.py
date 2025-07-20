import tiktoken
import torch

from llm_module.GPTModel import GPT_CONFIG_124M, GPTModel


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # greedy decoding
    # idx: current context index array, (batch, n_tokens)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # only focus on the last timestep
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


if __name__ == '__main__':
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)

    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)

