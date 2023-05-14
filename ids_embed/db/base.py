"""
Base Database for parsing/loading the IDS from a local file
"""
from ids_embed.db.vocab import build_vocabulary_from_scratch

from ids_embed.db.parse import eids2tree
from ids_embed.db.dgl_interface import IDSDataset
import sys, os

from ids_embed.db.parse import is_valid

sys.path.append(os.path.join(os.path.dirname(__file__), "parser"))

from ids_embed.utils import progress_log

import logging
import pickle


class GlyphDatabase:
    def __init__(self, eids_file, db_storage_path):
        self.open(eids_file, db_storage_path)

    def open(self, eids_file, db_storage_path):
        logging.info("eids DB open start")
        progress_log.update_push_stage("Opening eids DB")
        self.handle_path_path(eids_file, db_storage_path)
        self.load_cached_db(db_storage_path)

        if self.spec_db is None:
            self.read_spec_from_ids(eids_file, db_storage_path)

        logging.info("eids DB open okay")
        logging.info("vocabulary DB construct start")
        self.vocab_db = build_vocabulary_from_scratch(self.spec_db)
        logging.info("vocabulary DB construct okay")
        progress_log.update_pop_stage()

    def handle_path_path(self, eids_file, db_storage_path):
        self.eids_file = eids_file
        self.db_storage_path = db_storage_path
        self.spec_db_path = "{}/spec_db.pickle".format(db_storage_path)
        self.raw_spec_db_path = "{}/raw_spec_db.pickle".format(db_storage_path)

    def close(self):
        logging.info("closing db: saving cache db start")
        with open(self.spec_db_path, "wb") as handle:
            pickle.dump(self.spec_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.raw_spec_db_path, "wb") as handle:
            pickle.dump(self.raw_spec_db, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("closing db: saving cache db okay")

    def load_cached_db(self, db_storage_path):
        try:
            with open(self.spec_db_path, "rb") as handle:
                self.spec_db = pickle.load(handle)
            with open(self.raw_spec_db_path, "rb") as handle:
                self.raw_spec_db = pickle.load(handle)
            progress_log.update_total_task(
                progress_log.CATE_Parse_IDS_cache, total=len(self.spec_db)
            )
            progress_log.update_completed_task(
                progress_log.CATE_Parse_IDS_cache, inc=len(self.spec_db)
            )
        except:
            logging.info("No local stored result for spec db")
            self.spec_db = None
            self.raw_spec_db = None

    def read_spec_from_ids(self, eids_file, db_storage_path):
        self.spec_db = {}
        self.raw_spec_db = {}
        with open(eids_file, "r") as f:
            progress_log.update_total_task(
                progress_log.CATE_Parse_IDS, total=len(f.readlines())
            )
        with open(eids_file, "r") as f:
            for idx, l in enumerate(f):
                progress_log.update_completed_task(progress_log.CATE_Parse_IDS)
                if idx < 22:
                    continue
                name = l[1]
                if name == "?":
                    continue
                if l[3] != ";":
                    t = l[3:-1]
                    raw_ids = t
                    tree = eids2tree(raw_ids)
                    if tree is None or not is_valid(tree):
                        continue
                else:
                    tree = eids2tree(name)
                    if tree is not None:
                        # single radical glyph
                        raw_ids = tree.char
                    else:
                        raw_ids = ""

                self.spec_db[name] = tree
                self.raw_spec_db[name] = raw_ids

    def find_raw_ids(self, c):
        if c in self.raw_spec_db:
            ret = self.raw_spec_db[c]
            return ret
        else:
            return None

    def find(self, c):
        if c in self.spec_db:
            ret = self.spec_db[c]
            return ret
        else:
            return None

    def get_dgl_dataset(self):
        logging.info(
            "get IDS dataset: now we have {} valid entries".format(len(self.spec_db))
        )
        return IDSDataset(self)
