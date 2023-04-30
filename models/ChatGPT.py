import openai
import time

openai.api_key = ""
openai.api_base = "your_api_base_url"


class ChatGPT():
    def __init__(self):
        self.reset()

    def reset(self, instruction="You are a chatbot"):
        self.messages = [{"role": "system", "content": instruction}]
        self.history = []

    def __call__(self, prompt):
        time.sleep(20)
        self.messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )

        result = ''
        for choice in response.choices:
            result += choice.message.content

        self.messages.append({"role": "assistant", "content": result})
        self.history.append((prompt, result))

        return result

    def revoke(self, n=1):
        assert 0 <= n and n <= len(self.history)
        self.history = self.history[:-n]

    def force(self, new_reply):
        self.messages[-1]["content"] = new_reply
        self.history[-1] = (self.history[-1][0], new_reply, *self.history[-1][1:])
