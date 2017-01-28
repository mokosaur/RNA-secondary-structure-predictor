import random
import rna


class GeneticPredictor:
    def __init__(self, population_size=5, num_epoch=50):
        self.population_size = population_size
        self.num_epoch = num_epoch

    def predict(self, molecule):
        if not molecule.dot:
            molecule.dot = '.' * len(molecule.seq)
        population = [molecule] + [self.mutate(molecule) for i in range(self.population_size - 1)]
        for epoch in range(self.num_epoch):
            new_population = set(population)
            for i in range(self.population_size * 20):
                mutation = self.mutate(population[random.randrange(self.population_size)])
                new_population.add(mutation.repair())
            population = sorted(new_population, key=lambda x: x.evaluate())[-self.population_size:]

        for o in population:
            o.show()

    def crossover(self, a, b):
        pass

    def mutate(self, molecule):
        m = rna.pair_matrix(molecule)
        seq = molecule.seq
        dot = molecule.dot
        length = len(seq)
        x = random.randrange(length - 5)
        y = random.randrange(x + 5, length)
        if m[x, :].sum() == 0 and m[:, y].sum() == 0:
            dot = dot[:x] + '(' + dot[x + 1: y] + ')' + dot[y + 1:]
        if m[x, y] == 1:
            dot = dot[:x] + '.' + dot[x + 1: y] + '.' + dot[y + 1:]
        return rna.Molecule(seq, dot)
