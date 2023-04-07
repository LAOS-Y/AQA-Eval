from transformers import AutoModelForCausalLM, AutoTokenizer


class BLOOMZ():
    def __init__(self, name="bigscience/bloomz-560m", qa_prefix=True):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype="auto", device_map="auto"
        )
        self.qa_prefix = qa_prefix
        self.reset()

    def reset(self):
        self.context = ""
        self.last_reply = None

    def __call__(self, prompt, max_new_tokens=200):
        if self.qa_prefix:
            prompt = self.context + f"Q: {prompt}\n\nA: "
        else:
            prompt = self.context + f"{prompt}\n\n"

        input_ = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(input_, max_new_tokens=max_new_tokens)
        output = self.tokenizer.decode(output[0, input_.shape[1]:]).rstrip("</s>")
        self.context = f"{prompt}{output}\n\n"

        self.last_reply = output
        return output

    def rebuild_contexts(self, qa_list):
        context = ""
        for q, a in qa_list:
            context += f"Q: {q}\n\nA: {a}\n\n"

        self.context = context
        return context

    def teacher_force(self, new_reply):
        self.context = self.context[:-(len(self.last_reply) + 2)]
        self.context += f"{new_reply}\n\n"
        self.last_reply = new_reply
