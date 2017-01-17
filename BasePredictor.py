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
                        pos = dot.rfind(')')
                        a = sequence[pos]
                        sequence = rna.complementary(a).lower() + sequence
                        dot = '(' + dot
                    if repair and dot.count('(') != dot.count(')'):
                        continue
                    sequences.append(sequence.upper() if capitalize else sequence)
                    dots.append(dot)
            if num_chains == n_chains and len(sequences) == n_chains:
                result.append(sequences + dots)
        self.X = np.mat(result)
        return self.X


p = BasePredictor()
X = p.load_data("secondary.fa", capitalize=True, purify=True, repair=True)
# print(X)
# print(X.shape)

mol = X[0]
m = rna.Molecule(X[1,0], X[1,1])
m.show()
