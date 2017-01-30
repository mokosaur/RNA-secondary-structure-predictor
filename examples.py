# from BasePredictor import BasePredictor
# from NaivePredictor import NaivePredictor
import rna
# from GeneticPredictor import GeneticPredictor
from MFTPredictor import MFTPredictor

# p = BasePredictor()
# X = p.load_data("secondary.fa", capitalize=True, purify=True, repair=True)
# print(X)
# # print(X.shape)
#
# mol = X[0]
# m = rna.Molecule(X[1, 0], X[1, 1])
# m.show()
#
# p = NaivePredictor(22, substrings=False, max_examples=100, library='lasagne', data_model='matrix')
# p.load_data("secondary.fa", capitalize=True, purify=True, repair=True)
# p.train()
# print(p.predict("GGCCUGAGGAGACUCAGAAGCC"))
# print(p.predict("GGCCUGAGGAGACUCAGAAGCC"[::-1]))
# print(p.predict("CCCCUGAGGAGACUCAGAAGGG"))
# print(p.predict("AUCCUGAGGAGACUCAGAAGAU"))
# print(p.predict("AUCGUGAUGAGACUCAAAAGAU"))

# p = NaivePredictor(22, substrings=False, max_examples=100, library='lasagne')
# p.load_data("secondary.fa", capitalize=True, purify=True, repair=True)
# p.train()
# print(p.predict("GGCCUGAGGAGACUCAGAAGCC"))
# print(p.predict("GGCCUGAGGAGACUCAGAAGCC"[::-1]))
# print(p.predict("CCCCUGAGGAGACUCAGAAGGG"))
# print(p.predict("AUCCUGAGGAGACUCAGAAGAU"))

# p = GeneticPredictor(num_epoch=20)
# p.predict(rna.Molecule("GGCCUGAGGAGACUCAGAAGCC"))
# p.predict(rna.Molecule("GGCCCCAUCGUCUAGCGGUUAGGACGCGGCCCUCUCAAGGCCGAAACGGGGGUUCGAUUCCCCCUGGGGUCACCA"))


p = MFTPredictor(num_epoch=20)
p.load_data("secondary.fa", capitalize=True, purify=True, repair=True)
# print(p.predict(rna.Molecule("GGAAAACC")))
# p.train([rna.Molecule("GGCCUGAGGAGACUCAGAAGCC", "((((((((....)))))..)))")])
p.train()
result = p.predict(rna.Molecule("GGCCUGAGGAGACUCAGAAGCC"))
print(result)
result.show()
# result = p.predict(rna.Molecule("GGCCUGAGGAGACUCAGAAGCC"))
# print(p.predict(rna.Molecule("GGCCCCAUCGUCUAGCGGUUAGGACGCGGCCCUCUCAAGGCCGAAACGGGGGUUCGAUUCCCCCUGGGGUCACCA")))


# rna.pair_matrix(rna.Molecule("AUCGUGAUGAGACUCAAAAGAU", '.(((((.(.(....).))))))'), show=True)


# m = rna.Molecule('GGGAGCUCAACUCUCCCCCCCUUUUCCGAGGGUCAUCGGAACCA', '(((((.......))))).......(((((......)))))....')
# print(rna.pair_matrix(m, show=True))
# print(rna.complementarity_matrix(m, show=True))

# print(m.get_substrings(9))