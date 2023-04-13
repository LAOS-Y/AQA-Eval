from transformers import LlamaForCausalLM, LlamaTokenizer


class Llama():
    def __init__(self, tokenizer_path, model_path, qa_prefix=True):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
        self.qa_prefix = qa_prefix
        self.reset()

    def reset(self, init_context=""):
        self.init_context = init_context
        self.history = []

    @property
    def context(self):
        return self.init_context + self.rebuild_context(self.history)

    def __call__(self, prompt, max_new_tokens=200):
        if self.qa_prefix:
            full_prompt = self.context + f"Q: {prompt}\n\nA: "
        else:
            full_prompt = self.context + f"{prompt}\n\n"

        input_ = self.tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(input_, max_new_tokens=max_new_tokens)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(full_prompt):]

        self.history.append((prompt, output))
        return output

    def rebuild_context(self, qa_list):
        context = ""
        for q, a in qa_list:
            if q is not None:
                context += f"Q: {q}\n\n"
            if a is not None:
                context += f"A: {a}\n\n"

        return context

    def revoke(self, n=1):
        assert 0 <= n and n <= len(self.history)
        self.history = self.history[:-n]

    def teacher_force(self, new_reply):
        self.history[-1][1] = new_reply
