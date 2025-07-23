import torch

from extra.llm.LLMUtils import LLMUtils
from extra.llm.module.tokenizer.BPETokenizer import BPETokenizer
from extra.llm.gpt2.PretrainGPT2Utils import PretrainGPT2Utils

if __name__ == '__main__':

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BPETokenizer()

    gpt_model, model_config = PretrainGPT2Utils.build().to(device)
    gpt_model.eval()

    start_context = "Every effort moves you"
    torch.manual_seed(123)
    output_token_ids = LLMUtils.generate(
        model=gpt_model,
        idx=tokenizer.text_to_token_ids(start_context).to(device),
        max_new_tokens=50,
        context_size=LLMUtils.GPT_CONFIG_124M["context_length"],
        top_k=50,
        temperature=1.5
    )
    output_text = tokenizer.token_ids_to_text(output_token_ids)
    print(f"Output: {output_text}")
