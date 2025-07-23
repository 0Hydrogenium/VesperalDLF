import torch

from extra.llm.module.model.GPTModel import GPTModel
from extra.llm.module.tokenizer.BPETokenizer import BPETokenizer
from extra.llm.LLMUtils import LLMUtils


if __name__ == '__main__':

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer()
    model = GPTModel(LLMUtils.GPT_CONFIG_124M).to(device)

    file_path = "../../data/the_verdict/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as f:        text_data = f.read()

    text_characters_num = len(text_data)
    text_tokens_num = len(tokenizer.encode(text_data))
    print(f"characters num: {text_characters_num}")
    print(f"tokens num: {text_tokens_num}")

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = LLMUtils.create_dataloader_v1(
        train_data,
        tokenizer,
        batch_size=2,
        max_length=LLMUtils.GPT_CONFIG_124M["context_length"],
        stride=LLMUtils.GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = LLMUtils.create_dataloader_v1(
        val_data,
        tokenizer,
        batch_size=2,
        max_length=LLMUtils.GPT_CONFIG_124M["context_length"],
        stride=LLMUtils.GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    start_context = "Every effort moves you"

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = LLMUtils.train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=1,
        start_context=start_context,
        tokenizer=tokenizer
    )

    LLMUtils.plot_losses(
        num_epochs,
        tokens_seen,
        train_losses,
        val_losses
    )





