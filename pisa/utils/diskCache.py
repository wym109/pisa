
import os
import sqlite3

from utils import jsons

class DiskCache(object):
    def __init__(self, db_fpath):
        self.db_fpath = os.path.expandvars(os.path.expanduser(db_fpath))

    def __create_db(self, db_fpath):
        pass

    def load(self, obj):
        pass

    def store(self, obj):
        pass
