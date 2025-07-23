from extra.llm.module.tokenizer.BPETokenizer import BPETokenizer
from extra.llm.utils.TextGenerator import TextGenerator
from extra.llm.utils.ModelLoader import ModelLoader
from utils.GeneralTool import GeneralTool

if __name__ == '__main__':
    GeneralTool.set_global_seed(42)
    device = GeneralTool.device

    cfg_name = "PretrainGPT"
    cfg = GeneralTool.load_cfg(cfg_name)

    model_name = "GPT2-124M"
    model, cfg = ModelLoader.load(model_name, cfg)
    model = model.to(device)
    tokenizer = BPETokenizer()

    model.eval()

    input_text = "Every effort moves you"
    output_text = TextGenerator.generate_text(
        input_text,
        model,
        tokenizer,
        device,
        max_new_tokens=50,
        context_length=cfg["context_length"],
        temperature=1.4,
        top_k=25,
    )
    print(f"input_text: {input_text}")
    print(f"output_text: {output_text}")




