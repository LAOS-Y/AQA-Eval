class BSModel():
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.reset()

    def reset(self, init_context=""):
        self.if_start = False
        self.last_guess = None
        self.l = self.min
        self.r = self.max + 1

    def binary_search(self, prompt):
        if "bigger" in prompt:
            self.l = self.last_guess
        elif "smaller" in prompt:
            self.r = self.last_guess

        return (self.l + self.r) // 2

    def __call__(self, prompt):
        if prompt == "START":
            self.if_start = True
            self.last_guess = (self.l + self.r) // 2

        if self.if_start:
            self.last_guess = self.binary_search(prompt)
            return self.last_guess
