from extra.llm.module.text_generator.TextGenerator import TextGenerator


class InstructionTuningTextGenerator(TextGenerator):
    @classmethod
    def format_input(cls, entry):
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )
        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
        return instruction_text + input_text

    @classmethod
    def format_output(cls, entry):
        return f"\n\n### Response:\n{entry['output']}"
