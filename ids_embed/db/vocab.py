from typing import Dict, List
from ids_embed.utils.data_structure import flatten


class Vocabulary:

    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "?"
    IDC_tokens = "⿰⿱⿴⿵⿶⿷⿸⿹⿺⿻"  # Ideographic Description Characters

    _separator = "|"
    LABEL = "label"
    TOKEN = "token"
    NODE = "node"

    COUNT = 0

    def __init__(self):

        self._token_to_id: Dict[str, int] = dict()
        self._id_to_token: Dict[int, str] = dict()

        for t in [self.PAD, self.SOS, self.EOS, self.UNK]:
            self.add_token(t)
        for t in self.IDC_tokens:
            self.add_token(t)

    def add_token(self, token: str):

        if token in self._token_to_id:
            return
        self._id_to_token.update({self.COUNT: token})
        self._token_to_id.update({token: self.COUNT})
        self.COUNT += 1
        assert self.COUNT == len(self._token_to_id.keys())
        assert self.COUNT == len(self._id_to_token.keys())
        return

    def size(self):
        return self.COUNT

    @property
    def pad_idx(self):
        return self.token2id[self.PAD]

    @property
    def sos_idx(self):
        return self.token2id[self.SOS]

    @property
    def eos_idx(self):
        return self.token2id[self.EOS]

    @property
    def token2id(self) -> Dict[str, int]:
        return self._token_to_id

    @property
    def id2token(self) -> Dict[int, str]:
        return self._id_to_token

    @property
    def tokens(self) -> List[str]:
        return self._token_to_id.keys()


def build_vocabulary_from_scratch(spec_db):
    vocab = Vocabulary()
    for character in spec_db:
        token_list = [character] + [i.char for i in flatten(spec_db[character])]
        for t in token_list:
            vocab.add_token(t)

    return vocab
