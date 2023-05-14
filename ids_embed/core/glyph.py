"""
Basic data struct for representing a glyph (a component in a IDS)
"""


class GlyphSymbol:
    def __init__(self, char, position):
        self.char = char
        self.position = position

    def __str__(self):
        return "{}(pos:{})".format(self.char, self.position)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.char == other.char and self.position == other.position

    def __hash__(self):
        return hash((self.char, self.position))
