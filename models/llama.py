from transformers import LlamaForCausalLM, LlamaTokenizer


class Llama():
    def __init__(self, tokenizer_path, model_path, qa_prefix=True):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
        self.qa_prefix = qa_prefix
        self.reset()

    def reset(self):
        self.contexts = [""]
        self.last_reply = None

    def __call__(self, prompt, max_new_tokens=200):
        if self.qa_prefix:
            prompt = self.contexts[-1] + f"Q: {prompt}\n\nA: "
        else:
            prompt = self.contexts[-1] + f"{prompt}\n\n"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        outputs = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        prompt += f"{outputs}\n\n"

        self.last_reply = outputs
        self.contexts.append(prompt)
        return outputs

    def teacher_force(self, new_reply):
        self.contexts[-1] = self.contexts[-1][:-(len(self.last_reply) + 2)]
        self.contexts[-1] += f"{new_reply}\n\n"
        self.last_reply = new_reply
