import openai
import time

openai.api_key = ""

class ChatGPT():
    def __init__(self):
        self.messages = [{"role": "system", "content": "You are a chatbot"}]
    
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

        return result
    
    def reset(self):
        self.messages = [{"role": "system", "content": "You are a chatbot"}]
    
    def teacher_force(self, new_reply):
        self.messages.pop()

        self.messages.append({"role": "assistant", "content": new_reply})
        return