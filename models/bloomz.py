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
        self.contexts = [""]
        self.last_reply = None

    def __call__(self, prompt, max_new_tokens=200):
        if self.qa_prefix:
            prompt = self.contexts[-1] + f"Q: {prompt}\n\nA: "
        else:
            prompt = self.contexts[-1] + f"{prompt}\n\n"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens)
        outputs = self.tokenizer.decode(outputs[0, inputs.shape[1]:]).rstrip("</s>")
        prompt += f"{outputs}\n\n"

        self.last_reply = outputs
        self.contexts.append(prompt)
        return outputs

    def teacher_force(self, new_reply):
        self.contexts[-1] = self.contexts[-1][:-(len(self.last_reply) + 2)]
        self.contexts[-1] += f"{new_reply}\n\n"
        self.last_reply = new_reply
