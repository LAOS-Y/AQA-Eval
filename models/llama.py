from transformers import LlamaForCausalLM, LlamaTokenizer


class Llama():
    def __init__(self, tokenizer_path, model_path, qa_prefix=True):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto")
        self.qa_prefix = qa_prefix
        self.reset()

    def reset(self, init_context=""):
        self.context = init_context
        self.last_reply = None

    def __call__(self, prompt, max_new_tokens=200):
        if self.qa_prefix:
            prompt = self.context + f"Q: {prompt}\n\nA: "
        else:
            prompt = self.context + f"{prompt}\n\n"

        input_ = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(input_, max_new_tokens=max_new_tokens)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
        self.context = f"{prompt}{output}\n\n"

        self.last_reply = output
        return output

    def rebuild_context(self, qa_list):
        context = ""
        for q, a in qa_list:
            context += f"Q: {q}\n\nA: {a}\n\n"

        return context

    def teacher_force(self, new_reply):
        self.context = self.context[:-(len(self.last_reply) + 2)]
        self.context += f"{new_reply}\n\n"
        self.last_reply = new_reply
