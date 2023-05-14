"""
Transform the DB into a dgl batched graph dataset.
such that efficient training can be performed upon.
"""

from collections import namedtuple
from xml.dom.pulldom import CHARACTERS

import dgl
import torch
from torch.utils.data import DataLoader
from dgl.data import DGLDataset
import networkx as nx
import networkx as nx

from ids_embed.core.glyph import GlyphSymbol
from ids_embed.db.parse import is_valid
from ids_embed.utils.data_structure import flatten

import matplotlib.pyplot as plt


def plot_tree(g):
    nx_g = g.to_networkx()
    pos = nx.nx_agraph.graphviz_layout(nx_g, prog="dot")
    nx.draw(
        nx_g,
        pos,
        labels={idx: d for idx, d in enumerate(g.ndata["id"].tolist())},
        node_size=1000,
        #     nx.draw(g, pos, node_size=800,
        node_color=[[0.5, 0.5, 0.5]],
        arrowsize=30,
        font_size=12,
        font_color="whitesmoke",
    )
    plt.show()


def get_U_V(tree):
    if tree is None or isinstance(tree, GlyphSymbol) or len(tree) < 2:
        return [], []
    if len(tree) == 2:
        op, (operand1, operand2) = tree
        U1, V1 = get_U_V(operand1)
        U2, V2 = get_U_V(operand2)

        op1_position = (
            operand1.position
            if isinstance(operand1, GlyphSymbol)
            else operand1[0].position
        )
        op2_position = (
            operand2.position
            if isinstance(operand2, GlyphSymbol)
            else operand2[0].position
        )
        op_position = op.position

        return U1 + U2 + [op1_position, op2_position], V1 + V2 + [
            op_position,
            op_position,
        ]


def get_dgl_graph(tree, db):
    id_mapping = {
        idx: db.vocab_db.token2id[s.char] for idx, s in enumerate(flatten(tree))
    }

    U, V = get_U_V(tree)
    U_ = [u for u in U]
    V_ = [v for v in V]
    g = dgl.graph(
        (torch.tensor(U_, dtype=torch.long), torch.tensor(V_, dtype=torch.long))
    )

    node_features = torch.zeros(len(id_mapping), dtype=torch.long)
    for i, idx in id_mapping.items():
        node_features[i] = idx
    g.ndata["id"] = node_features
    return g


def get_batched_dgl_graph(db):
    trees = []
    glyph_ids = []
    characters = []
    for c in db.spec_db:
        # try:
        spec = db.find(c)
        if not is_valid(spec) or len(flatten(spec)) < 2:
            continue
        g = get_dgl_graph(spec, db)
        trees.append(g)
        glyph_ids.append(flatten(spec))
        characters.append(c)
        # except:
        #     continue

    return trees, glyph_ids, characters


class IDSDataset(DGLDataset):
    def __init__(self, db):
        self.db = db
        super().__init__(name="ids")

    def process(self):
        trees, seq_labels, characters = get_batched_dgl_graph(self.db)
        self.trees = trees
        self.characters = torch.tensor(
            [self.db.vocab_db.token2id[c] for c in characters]
        )

        max_len = max([len(i) for i in seq_labels])
        self.labels = (
            torch.zeros(len(seq_labels), max_len + 2, dtype=torch.long)
            + self.db.vocab_db.pad_idx
        )
        for row_idx, ids in enumerate(seq_labels):
            length = len(ids)
            flush_data = (
                [self.db.vocab_db.sos_idx]
                + [self.db.vocab_db.token2id[i.char] for i in ids]
                + [self.db.vocab_db.eos_idx]
            )
            for curr in range(len(flush_data)):
                self.labels[row_idx, curr] = flush_data[curr]
        return

    def __getitem__(self, i):
        return self.trees[i], self.labels[i], self.characters[i]

    def __len__(self):
        return len(self.trees)

    def get_data_loader(self):
        IDSBatch = namedtuple("IDSBatch", ["tree", "label", "character"])

        def batcher(dev):
            def batcher_dev(batch):
                trees, labels, characters = zip(*batch)
                characters = torch.tensor(characters)
                batched_trees = dgl.batch(trees)
                batch_num_nodes = batched_trees.batch_num_nodes()
                max_len = max(batch_num_nodes)
                batched_labels = (
                    torch.zeros(batch_num_nodes.shape[0], max_len + 2, dtype=torch.long)
                    + self.db.vocab_db.pad_idx
                )
                for row_idx, ids in enumerate(labels):
                    ids = ids.tolist()
                    start = ids.index(self.db.vocab_db.sos_idx)
                    end = ids.index(self.db.vocab_db.eos_idx)
                    flush_data = (
                        [self.db.vocab_db.sos_idx]
                        + ids[start + 1 : end]
                        + [self.db.vocab_db.eos_idx]
                    )
                    for curr in range(len(flush_data)):
                        batched_labels[row_idx, curr] = flush_data[curr]

                return IDSBatch(
                    tree=batched_trees, label=batched_labels.t(), character=characters
                )

            return batcher_dev

        device = torch.device("cpu")
        return DataLoader(
            dataset=self,
            # batch_size=64,
            batch_size=128,
            collate_fn=batcher(device),
            shuffle=True,
            num_workers=0,
        )


def encode_spec(spec, db, model):
    """
    input a spec string. output the embedding of a graph
    """
    g = get_dgl_graph(spec, db)
    batched_trees = dgl.batch([g])
    graph_embedding = model.encode(batched_trees, return_embed=True)
    return graph_embedding


def encode_spec_batch(specs, db, model):
    """
    input a spec string. output the embedding of a graph
    """
    gs = [get_dgl_graph(spec, db) for spec in specs]
    batched_trees = dgl.batch(gs)
    graph_embedding = model.encode(batched_trees, return_embed=True)
    return graph_embedding
