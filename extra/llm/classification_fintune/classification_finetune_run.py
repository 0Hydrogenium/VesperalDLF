import torch
import pandas as pd
from torch.utils.data import DataLoader

from extra.llm.module.tokenizer.BPETokenizer import BPETokenizer
from extra.llm.classification_fintune.SpamDataset import SpamDataset
from extra.llm.gpt2.PretrainGPT2Utils import PretrainGPT2Utils

if __name__ == '__main__':

    """
        classification finetune
        - drawback: can only predict the categories that model encounter during training
    """

    random_seed = 123

    data_path = "../../../data/sms_spam_collection/SMSSpamCollection.csv"
    df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])
    print(df["label"].value_counts())

    # downsampling data to balance
    num_spam = df[df["label"] == "spam"].shape[0]
    ham_subset = df[df["label"] == "ham"].sample(num_spam, random_state=random_seed)
    df = pd.concat([ham_subset, df[df["label"] == "spam"]])
    print(df["label"].value_counts())

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    train_frac, validation_frac = 0.7, 0.1
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    tokenizer = BPETokenizer()

    train_dataset = SpamDataset(train_df, tokenizer)
    print(f"train data max length: {train_dataset.max_length}")
    val_dataset = SpamDataset(validation_df, tokenizer)
    print(f"val data max length: {val_dataset.max_length}")
    test_dataset = SpamDataset(test_df, tokenizer)
    print(f"test data max length: {test_dataset.max_length}")

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer()

    gpt_model, model_config = PretrainGPT2Utils.build().to(device)
    gpt_model.eval()

    """
        add classification head
    """

    # freeze all layer params
    for param in gpt_model.parameters():
        param.requires_grad = False

    torch.manual_seed(random_seed)
    num_classes = 2
    gpt_model.num_head = torch.nn.Linear(in_features=model_config["emb_dim"], out_features=num_classes)

    # make the output layer, final LayerNorm, and the last Transformer block trainable
    for param in gpt_model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in gpt_model.final_norm.parameters():
        param.requires_grad = True
    for param in gpt_model.num_head.parameters():
        param.requires_grad = True

    # only focus on the last line token
    # the last token aggregates all the previous tokens info





