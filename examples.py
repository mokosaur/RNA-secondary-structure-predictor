from BasePredictor import BasePredictor
from NaivePredictor import NaivePredictor
import rna

p = BasePredictor()
X = p.load_data("secondary.fa", capitalize=True, purify=True, repair=True)
# print(X)
# # print(X.shape)
#
# mol = X[0]
# m = rna.Molecule(X[1, 0], X[1, 1])
# m.show()

p = NaivePredictor(22, substrings=False, max_examples=100, library='lasagne')
p.load_data("secondary.fa", capitalize=True, purify=True, repair=True)
p.train()
print(p.predict("GGCCUGAGGAGACUCAGAAGCC"))

# m = rna.Molecule('GGGAGCUCAACUCUCCCCCCCUUUUCCGAGGGUCAUCGGAACCA', '(((((.......))))).......(((((......)))))....')
# print(rna.pair_matrix(m, show=True))
# print(rna.complementarity_matrix(m, show=True))

# print(m.get_substrings(9))