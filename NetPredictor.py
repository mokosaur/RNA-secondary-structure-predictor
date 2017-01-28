import random
import rna
import math
import numpy as np


class NetPredictor:
    class Neuron:
        def __init__(self, i, j):
            self.i = i
            self.j = j

    def __init__(self, num_epoch=10, alpha=2, beta=2, gamma=2, mi=2, ni=5):
        self.num_epoch = num_epoch
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mi = mi
        self.ni = ni
        self.t = 1

    def predict(self, molecule):
        if not molecule.dot:
            molecule.dot = '.' * len(molecule.seq)
        self.seq = molecule.seq
        self.n = len(molecule.seq)
        self.neurons = rna.pair_matrix(molecule) + np.random.uniform(-1, 1, (self.n, self.n))
        for i in range(self.num_epoch):
            self.epoch()
            self.t += (self.t) / (i + 1)
            np.set_printoptions(precision=2)
            print(np.triu(self.neurons))

        dot = molecule.dot
        for x in range(len(self.neurons)):
            for y in range(x + 1, len(self.neurons)):
                if self.node_weight(x, y) == self.ni:
                    if self.neurons[x, y] > 0:
                        print(self.node_weight(x, y), self.seq[x], self.seq[y], 'comp')
                        dot = dot[:x] + '(' + dot[x + 1: y] + ')' + dot[y + 1:]
                if self.node_weight(x, y) == self.ni / 2:
                    if self.neurons[x, y] > 0.4:
                        print(self.seq[x], self.seq[y], 'GU')
                        dot = dot[:x] + '(' + dot[x + 1: y] + ')' + dot[y + 1:]
        return rna.Molecule(molecule.seq, dot)

    def epoch(self):
        beta = 1 / self.t
        neurons = list(range(self.n * (self.n - 1) // 2))
        random.shuffle(neurons)
        for k in neurons:
            x = self.n - 2 - math.floor(math.sqrt(-8 * k + 4 * self.n * (self.n - 1) - 7) / 2.0 - 0.5)
            y = int(k + x + 1 - self.n * (self.n - 1) / 2 + (self.n - x) * ((self.n - x) - 1) / 2)
            sum = 0
            for i in range(len(self.neurons)):
                for j in range(i + 1, len(self.neurons)):
                    if x != i and y != j:
                        sum += self.neurons[i, j] * self.weight(x, y, i, j)
            sum += self.node_weight(x, y)
            print(x, y, sum, self.neurons[x, y], (math.tanh(beta * sum) + 1) / 2)
            self.neurons[x, y] = math.tanh(beta * sum)

            # for x in range(len(self.neurons)):
            #     for y in range(x + 1, len(self.neurons)):
            #         sum = 0
            #         for i in range(len(self.neurons)):
            #             for j in range(i + 1, len(self.neurons)):
            #                 if x != i and y != j:
            #                     sum += self.neurons[i, j] * self.weight(x, y, i, j)
            #         sum += self.node_weight(x, y)
            #         self.neurons[x, y] = math.tanh(beta * sum)

    def weight(self, r, c, i, j):
        e = 0
        if r == i:
            e -= self.alpha
        if c == j:
            e -= self.beta
        if r < i < c < j or i < r < j < c:
            e -= self.gamma
        if i == r + 1 and j == c - 1:
            e += self.mi
        if i == r + 2 and j == c - 2:
            e += self.mi
        if i == r + 3 and j == c - 3:
            e += self.mi
        return e

    def node_weight(self, r, c):
        e = 0
        if self.seq[r] == rna.complementary(self.seq[c]):
            e += self.ni
        elif self.seq[r] + self.seq[c] in ['GU', 'UG']:
            e += self.ni / 2
        else:
            e -= self.ni
        return e
