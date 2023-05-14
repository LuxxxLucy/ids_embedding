from difflib import context_diff
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "antlr"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import antlr4
from antlr4 import *
from idsLexer import idsLexer
from idsParser import idsParser
from idsVisitor import idsVisitor


class SimpleIdsVistor(idsVisitor):
    """
    this vistor simply returns a list of lists of the tree structure
    """

    def __init__(self, font, eidsDB, config):
        super().__init__()
        self.font = font
        self.eidsDB = eidsDB
        self.config = config

    # Visit a parse tree produced by idsParser#glyph.
    def visitGlyph(self, ctx: idsParser.GlyphContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by idsParser#CJKPART.
    def visitCJKPART(self, ctx: idsParser.CJKPARTContext):
        # return self.visitChildren(ctx)
        c = ctx.getText()
        t = ctx.getSourceInterval()

        r = GlyphSymbol(c, t)

        return r

    # Visit a parse tree produced by idsParser#CJKBinaryComposition.
    def visitCJKBinaryComposition(self, ctx: idsParser.CJKBinaryCompositionContext):
        if ctx.indicator is not None:
            """for now we ignore the indicator"""
            indicator = ctx.indicator.text
            c = ctx.indicator.tokenIndex
            r = GlyphSymbol(indicator, c)
            return r
        else:
            op1 = self.visit(ctx.op1)
            op2 = self.visit(ctx.op2)
            op = ctx.op.text
            # return self.visitChildren(ctx)
            return [op, [op1, op2]]

    # Visit a parse tree produced by idsParser#CJKTrinaryComposition.
    def visitCJKTrinaryComposition(self, ctx: idsParser.CJKTrinaryCompositionContext):
        op1 = self.visit(ctx.op1)
        op2 = self.visit(ctx.op2)
        op3 = self.visit(ctx.op3)
        op = ctx.op.text
        t = ctx.getSourceInterval()
        op = GlyphSymbol(op, t)
        # return self.visitChildren(ctx)
        return [op, [op1, op2, op3]]


def get_tree(text, font=None, eidsDB=None, config=None):
    input = InputStream(text)
    lexer = idsLexer(input)
    stream = CommonTokenStream(lexer)
    parser = idsParser(stream)
    tree = parser.glyph()
    visitor = SimpleIdsVistor(font, eidsDB, config)
    obj = visitor.visit(tree)
    return obj
