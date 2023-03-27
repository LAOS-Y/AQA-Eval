from transformers import AutoModelForCausalLM, AutoTokenizer
class BLOOMZ():
    def __init__(self, ckpt_name="bigscience/bloomz-560m", qa_prefix=True):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            ckpt_name, torch_dtype="auto", device_map="auto"
        )
        self.qa_prefix = qa_prefix
        self.reset()
    def reset(self):
        self.contexts = [""]
    def __call__(self, prompt, max_new_tokens=200):
        if self.qa_prefix:
            prompt = self.contexts[-1] + f"Q: {prompt}\n\nA: "
        else:
            prompt = self.contexts[-1] + f"{prompt}\n\n"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        outputs = self.tokenizer.decode(outputs[0, inputs.shape[1]:]).rstrip("</s>")
        prompt += f"{outputs}\n\n"
        self.contexts.append(prompt)
        return outputs