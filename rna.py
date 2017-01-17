import webbrowser


class Molecule:
    def __init__(self, seq, dot=None):
        self.__seq = seq
        if dot:
            self.__dot = dot

    def show(self):
        if self.__dot:
            webbrowser.open(
                "http://nibiru.tbi.univie.ac.at/forna/forna.html?id=url/name&sequence={}&structure={}".format(
                    self.__seq,
                    self.__dot))
        else:
            raise Exception('Structure notation does not exist.')

    def set_dot(self, dot):
        self.__dot = dot


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
