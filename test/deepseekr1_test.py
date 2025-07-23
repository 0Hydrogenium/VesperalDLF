from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

from utils.GeneralTool import GeneralTool

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = f"{GeneralTool.root_path}/model/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    model = AutoModelForCausalLM.from_pretrained(model_save_path).to(device)

    t0 = time.time()
    user_input = "你好"
    messages = [{
        "role": "user",
        "content": user_input
    }]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(device)
    attn_mask = torch.ones(inputs.shape, dtype=torch.long).to(device)

    outputs = model.generate(
        inputs,
        attention_mask=attn_mask,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output_text)
    t1 = time.time()
    print(f"time: {t1 - t0:.4f}s")


