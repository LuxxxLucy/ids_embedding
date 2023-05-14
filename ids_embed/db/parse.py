import imp

import logging
import traceback

from ids_embed.core.glyph import GlyphSymbol
from ids_embed.utils.data_structure import flatten

from antlr4 import *
from antlr4.tree.Trees import Trees
from ids_embed.parser.antlr.idsLexer import idsLexer
from ids_embed.parser.antlr.idsListener import idsListener
from ids_embed.parser.antlr.idsVisitor import idsVisitor
from ids_embed.parser.antlr.idsParser import idsParser


class ParseIdsVistor(idsVisitor):
    """
    this vistor simply returns a list of lists of the tree structure
    """

    def __init__(self):
        super().__init__()

    def counter_add(self):
        self.counter += 1

    # Visit a parse tree produced by idsParser#glyph.
    def visitGlyph(self, ctx: idsParser.GlyphContext):
        self.counter = -1
        return self.visitChildren(ctx)

    # Visit a parse tree produced by idsParser#CJKPART.
    def visitCJKPART(self, ctx: idsParser.CJKPARTContext):
        r = ctx.getText()
        self.counter_add()
        return GlyphSymbol(r, self.counter)

    # Visit a parse tree produced by idsParser#CJKBinaryComposition.
    def visitCJKBinaryComposition(self, ctx: idsParser.CJKBinaryCompositionContext):
        # todo: resolve the error here
        # if ctx.indicator is not None:
        #     ''' for now we ignore the indicator'''
        #     indicator  = ctx.indicator.text
        #     return indicator
        if ctx.op is None:
            logging.info("parsing failed in visitCJKbinaryComposition. ctx.op is None")
            return
        else:
            self.counter_add()
            op = GlyphSymbol(ctx.op.text, self.counter)
            op1 = self.visit(ctx.op1)
            op2 = self.visit(ctx.op2)
            return [op, [op1, op2]]

    # Visit a parse tree produced by idsParser#Unknown.
    def visitUnknown(self, ctx: idsParser.UnknownContext):
        r = "?"
        self.counter_add()
        return GlyphSymbol(r, self.counter)

    # Visit a parse tree produced by idsParser#CJKTrinaryComposition.
    def visitCJKTrinaryComposition(self, ctx: idsParser.CJKTrinaryCompositionContext):
        op1 = self.visit(ctx.op1)
        op2 = self.visit(ctx.op2)
        op3 = self.visit(ctx.op3)
        op = ctx.op.text
        self.counter_add()
        op = GlyphSymbol(op, self.counter)
        # return self.visitChildren(ctx)
        return [op, [op1, op2, op3]]


def eids2tree(text):
    text = "Glyph test {}".format(text)
    input = InputStream(text)
    lexer = idsLexer(input)
    stream = CommonTokenStream(lexer)
    parser = idsParser(stream)
    tree = parser.glyph()
    visitor = ParseIdsVistor()
    obj = visitor.visit(tree)
    return obj


def is_valid(tree):
    def is_valid_node(n):
        return n is not None and isinstance(n, GlyphSymbol)

    return all(list(map(is_valid_node, flatten(tree))))
