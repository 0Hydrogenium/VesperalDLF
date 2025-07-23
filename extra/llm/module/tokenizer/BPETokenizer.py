import tiktoken
import torch


class BPETokenizer:
    def __init__(self):
        self.tiktoken = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        return self.tiktoken.encode(text)

    def text_to_token_ids(self, text):
        encoded = self.tiktoken.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0)  # remove batch dimension
        return self.tiktoken.decode(flat.tolist())
