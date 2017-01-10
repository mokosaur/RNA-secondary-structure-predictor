import numpy as np


class RNAPredictor:
    def __init__(self):
        self.X = np.zeros((0, 2))

    def load_data(self, filename, n_chains=1, capitalize=False, purify=False):
        with open(filename) as file:
            data = file.read()
        data = data.split('\n\n')
        result = []
        for line in data:
            chains = line.split(">")
            num_chains = 0
            sequences = []
            dots = []
            for chain in chains:
                if "model:1/" in chain:
                    num_chains += 1
                    splitted = chain.split('\n')
                    sequences.append(splitted[1].upper() if capitalize else splitted[1])
                    dots.append(splitted[2].replace('[', '(').replace(']', ')').replace('-', '.') if purify else splitted[2])
            if num_chains == n_chains:
                result.append(sequences + dots)
        self.X = np.mat(result)
        return self.X


p = RNAPredictor()
X = p.load_data("secondary.fa", capitalize=True, purify=True)
print(X)
print(X.shape)
