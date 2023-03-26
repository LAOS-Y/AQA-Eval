import random

class Evaluator():
    def __init__(self, min=0, max=100):
        self.min = min
        self.max = max
            
    def init_model(self, model, verbose=False):
        prompt = "You are required to guess the random number which I have just picked between {} and {}.\n" \
                 "I will only give responses such as 'The true number is bigger than this guess' or 'The true number is smaller than this guess' or 'The true number is equal to this guess'.\n" \
                 "Adjust your guess according to my response.\n" \
                 "Try as few times as you can.\n" \
                 "Start guessing after receiving 'START' command.\n" \
                 "Stop guessing after receiving 'STOP' command.\n" \
                 "Reply 'OK' if you understand.".format(self.min, self.max)
        if verbose:
            print("Q: {}".format(prompt))

        reply = model(prompt)
        if verbose:
            print("A: {}".format(reply))
        
        return reply == "OK"
    
    def test_one_time(self, model, verbose=False):
        if not self.init_model(model, verbose):
            return
        
        guess = None
        cnt = 0
        target = random.randint(self.min, self.max)
        prompt = "START"
        
        if verbose:
            print("Picked Random Number: {}".format(target))
        
        while guess != target:
            if verbose:
                print("Q: {}".format(prompt))
            
            guess = model(prompt)
            cnt += 1
            if verbose:
                print("A: {}".format(guess))

            try:
                guess = int(guess)
            except ValueError:
                return

            
            if guess < target:
                prompt = "The true number is bigger than this guess"
            elif guess > target:
                prompt = "The true number is smaller than this guess"
            else:
                prompt = "The true number is equal to this guess"

        return cnt