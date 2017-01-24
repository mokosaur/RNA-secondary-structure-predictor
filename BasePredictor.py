import numpy as np
import rna


class BasePredictor:
    def __init__(self):
        self.X = np.zeros((0, 2))

    def load_data(self, filename, n_chains=1, capitalize=False, purify=False, repair=False):
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
                    dot = splitted[2].replace(
                        '[', '.').replace(']', '.').replace('<', '.').replace('>', '.').replace('{', '.').replace(
                        '}', '.').replace('-', '.') if purify else splitted[2]
                    sequence = splitted[1]
                    if repair and dot.count('(') + 1 == dot.count(')'):  # only for single-stranded?
                        dot = '(' + dot
                        pos = rna.match_parentheses(dot, 0) - 1
                        a = sequence[pos]
                        sequence = rna.complementary(a).lower() + sequence
                    if repair and dot.count('(') != dot.count(')'):
                        continue
                    sequences.append(sequence.upper() if capitalize else sequence)
                    dots.append(dot)
            if num_chains == n_chains and len(sequences) == n_chains:
                result.append(sequences + dots)
        self.X = np.mat(result)
        return self.X

    def train(self, X=None):
        if X is None and self.X.shape[0] == 0:
            raise Exception('There is no data to train.')
        else:
            if X is not None:
                self.X = X
            self.train_X()

    def train_X(self):
        raise Exception("You cannot train a base predictor.")

    def predict(self, seq):
        raise Exception("You cannot predict with a base predictor.")
