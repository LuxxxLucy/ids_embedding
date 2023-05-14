from ids_embed.embedding.trees import TreeLSTM_Model
from ids_embed.embedding.loss import (
    SequenceCrossEntropyLoss,
)

import os
from ids_embed.db.base import GlyphDatabase as DB
import logging
from tqdm import tqdm
import torch
import numpy as np

from sklearn.neighbors import NearestNeighbors
from ids_embed.db.dgl_interface import encode_spec, encode_spec_batch
from ids_embed.db.parse import eids2tree

__all__ = ["ids_embedding_runner"]


class ids_embedding_runner:
    def __init__(self, args, config):
        self.args = args
        self.config = config

        output_path = config.storage_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path

        self.prepare()

    def prepare(self):
        self.db = DB(self.config.eidsDB_path, self.config.storage_path)

        vocab_db = self.db.vocab_db
        pad_idx = vocab_db.pad_idx  # Used for padding short sentences
        sos_idx = vocab_db.sos_idx  # Start-of-sentence token
        eos_idx = vocab_db.eos_idx  # End-of-sentence token
        num_vocabs = vocab_db.size()

        self.model = TreeLSTM_Model(
            self.db, self.config, num_vocabs, pad_idx, sos_idx, eos_idx
        )
        self.db.close()

    def train(self):
        self.load_net()

        IDS_dataset = self.db.get_dgl_dataset()
        data_loader = IDS_dataset.get_data_loader()

        # create the optimizer
        optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )

        loss_fn = SequenceCrossEntropyLoss(
            self.db.vocab_db.pad_idx, reduction="batch-mean"
        )
        IDC_tokens = self.db.vocab_db.IDC_tokens
        important_idxs = [self.db.vocab_db.token2id[i] for i in IDC_tokens]

        def acc_fn(pred, label):
            mask = label != self.db.vocab_db.pad_idx
            t = torch.eq(label * 1.0, pred.argmax(-1) * 1.0)
            acc = float(torch.sum(t * mask.float()) / (mask.sum()))
            return acc

        self.model.train()
        logging.info("start training")
        with tqdm(range(self.config.train.n_epochs), unit="step") as tsteps:
            for t in tsteps:
                for step, (g, labels, _) in enumerate(data_loader):
                    logits, embedding_states = self.model(g, labels)
                    pred = logits
                    loss_pred = loss_fn(logits, labels)
                    regularization = (embedding_states**2).mean()

                    token_embedding = self.model.embedding.weight
                    token_embedding_regularization = (token_embedding**2).mean()

                    loss = (
                        loss_pred
                        + regularization * 1
                        + token_embedding_regularization * 1
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    acc = acc_fn(pred, labels)

                    tsteps.set_postfix(
                        message="Epoch {:02d} | Step {:02d} | Loss {:.4f} | Loss_pred {:.3f} | Reg {:.3f} | Embed_reg {:.3f} | acc {:.4f} |".format(
                            t,
                            step,
                            loss.item(),
                            loss_pred.item(),
                            regularization.item(),
                            token_embedding_regularization.item(),
                            acc,
                        )
                    )
                logging.info("epoch {}".format(t))
                self.save_net()
                self.test()
        logging.info("training end")
        self.save_net()
        self.test()

    def load_net(self):
        checkpoint_path = os.path.join(self.output_path, "embedding_model.pth")
        if os.path.exists(checkpoint_path):
            logging.info("load old net from {}".format(checkpoint_path))
            self.model.load_state_dict(torch.load(checkpoint_path))
        else:
            logging.info("initialize new net in {}".format(checkpoint_path))

    def save_net(self):
        checkpoint_path = os.path.join(self.output_path, "embedding_model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)

    def test(self):
        self.load_net()
        self.model.eval()
        IDS_dataset = self.db.get_dgl_dataset()
        data_loader = IDS_dataset.get_data_loader()

        embeds = []
        ids = []
        db = self.db
        for g, t, cs in tqdm(data_loader):
            e = self.model.encode(g, return_embed=True)
            embeds.append(e)
            ids.append(cs)
        embeds = torch.cat(embeds).detach().numpy()
        ids = torch.cat(ids).detach().numpy().tolist()

        X = np.array(embeds)
        nbrs = NearestNeighbors(n_neighbors=10, algorithm="ball_tree").fit(X)
        distances, indices = nbrs.kneighbors(X)

        token = "亨"
        tree = db.find(token)
        target_embedding = encode_spec_batch([tree], db, self.model).detach().numpy()
        _, indices = nbrs.kneighbors(target_embedding)
        similar_tokens = list(
            map(lambda a: self.db.vocab_db.id2token[ids[a]], indices[0])
        )
        id = self.db.vocab_db.token2id[token]
        logging.info("the tokein is {}".format(token))
        print("similar context {}".format(" ".join(similar_tokens)))

        test_ids = "⿱⿰耳口之"
        tree = eids2tree(test_ids)
        target_embedding = encode_spec_batch([tree], db, self.model).detach().numpy()
        _, indices = nbrs.kneighbors(target_embedding)
        similar_tokens = list(
            map(lambda a: self.db.vocab_db.id2token[ids[a]], indices[0])
        )
        print("for ids {}".format(test_ids))
        print("similar context {}".format(" ".join(similar_tokens)))

        return
