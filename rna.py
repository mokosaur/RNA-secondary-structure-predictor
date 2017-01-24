import webbrowser
import numpy as np
import matplotlib.pyplot as plt


class Molecule:
    def __init__(self, seq, dot=None):
        self.__seq = seq
        self.__dot = None
        if dot:
            self.__dot = dot

    def __repr__(self):
        return "\n{}\n{}\n".format(self.seq, self.dot)

    def __str__(self):
        return "{}".format(self.seq)

    def show(self):
        if self.__dot:
            webbrowser.open(
                "http://nibiru.tbi.univie.ac.at/forna/forna.html?id=url/name&sequence={}&structure={}".format(
                    self.__seq,
                    self.__dot))
        else:
            raise Exception('Structure notation does not exist.')

    @property
    def seq(self):
        return self.__seq

    @seq.setter
    def seq(self, seq):
        self.__seq = seq

    @property
    def dot(self):
        return self.__dot

    @dot.setter
    def dot(self, dot):
        self.__dot = dot

    def get_substrings(self, length):
        if self.dot is None:
            raise Exception("There is no structure given for this molecule.")
        else:
            valid = []
            for i in range(len(self.seq) - length + 1):
                substring = self.seq[i:i + length]
                subdot = self.dot[i:i + length]
                ctr = 0
                for j in range(length):
                    if subdot[j] == '(':
                        ctr += 1
                    if subdot[j] == ')':
                        ctr -= 1
                    if ctr < 0:
                        break
                if ctr == 0:
                    valid.append(Molecule(substring, subdot))
                # if subdot.count('(') == subdot.count(')'):
                #     valid.append(Molecule(substring, subdot))
            return valid


def complementary(a):
    a = a.upper()
    if a == 'A':
        return 'U'
    if a == 'U':
        return 'A'
    if a == 'C':
        return 'G'
    if a == 'G':
        return 'C'
    raise Exception('The given letter is not a valid RNA base.')


def encode_rna(x):
    return [0 if y == 'A' else 1 if y == 'U' else 2 if y == 'G' else 3 for y in x]


def match_parentheses(dot, position):
    stack = 0
    for i in range(position + 1, len(dot)):
        if dot[i] == '(':
            stack += 1
        elif dot[i] == ')':
            if stack == 0:
                return i
            else:
                stack -= 1
    return -1


def pair_matrix(m, show=False):
    l = len(m.seq)
    p = np.zeros((l, l))
    dot = m.dot
    for begin in range(l):
        if dot[begin] == '(':
            end = match_parentheses(dot, begin)
            p[begin, end] = p[end, begin] = 1

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(p, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_xticks(np.arange(l))
        ax.set_yticks(np.arange(l))
        ax.set_xticklabels([i for i in m.seq])
        ax.set_yticklabels([i for i in m.seq])
        plt.show()

    return p


def complementarity_matrix(m, show=False):
    l = len(m.seq)
    p = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if m.seq[i] == complementary(m.seq[j]):
                p[i, j] = 2
            if m.seq[i] == 'G' and m.seq[j] == 'U' or m.seq[i] == 'U' and m.seq[j] == 'G':
                p[i, j] = 1

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(p, interpolation='nearest')
        ax.set_xticks(np.arange(l))
        ax.set_yticks(np.arange(l))
        ax.set_xticklabels([i for i in m.seq])
        ax.set_yticklabels([i for i in m.seq])
        plt.show()

    return p
