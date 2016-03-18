
import os
import time
import sqlite3

from utils import jsons

sqlite3.register_converter('JSON', jsons.loads)


class DiskCache(object):
    TABLE_SCHEMA = (
        'CREATE TABLE IF NOT EXISTS cache('
            'hash INTEGER PRIMARY KEY,'
            'data JSON,'
            'accesstime INTEGER)'
        'WITHOUT ROWID;')

    def __init__(self, db_fpath, row_limit=1000):
        self.__db_fpath = os.path.expandvars(os.path.expanduser(db_fpath))
        self.__instantiate_db()
        assert 0 < row_limit < 1e6, 'Invalid row_limit:' + str(row_limit)
        self.__row_limit = row_limit

    def __instantiate_db(self):
        exists = True if os.path.isfile(self.db_fpath) else False
        self.conn = sqlite3.connect(
            self.__db_fpath,
            detect_types=sqlite3.PARSE_DECLTYPES|sqlite3.PARSE_COLNAMES
        )
        self.c = self.conn.cursor()
        c.row_factory = sqlite3.Row
        if exists:
            # check that the table format is valid
            schema = self.conn.execute('.schema cache')
            if schema.strip().lower() != self.TABLE_SCHEMA.strip().lower():
                raise ValueError(
                    'Existing database at "%s" has non-matching schema:\n'
                    '"""%s"""' % (self.__db_fpath, schema)
                )
        else:
            # Create the table for storing (hash, data, timestamp) tuples
            self.conn.execute(self.TABLE_SCHEMA)

    def __now_millisec_int(self):
        return int(time.time() * 1000)

    def load(self, hash_val):
        assert isinstance(hash_val, int)
        rows = self.c.execute('UPDATE cache SET accesstime=(?) WHERE hash=?;',
                              (self.__now_millisec_int(), hash_val))
        rows = self.c.execute('SELECT data FROM cache WHERE hash=(?)';,
                              hash_val)
        return rows.fetchone()['data']

    def store(self, hash_val, obj):
        assert isinstance(hash_val, int)
        js = jsons.dumps(obj)
        self.conn.execute('INSERT INTO cache VALUES (?, ?, ?);',
                          (hash_val, js, self.__now_millisec_int()))

        count = self.conn.execute('SELECT COUNT (*) FROM cache;')
        n_to_remove = count - self.__row_limit
        if n_to_remove <= 0:
            return

        # Find the access time of the n-th element (sorted ascending by access
        # time)
        nth_access_time = self.conn.execute(
            'SELECT accesstime FROM cache '
            'ORDER BY accesstime ASC LIMIT 1 OFFSET ?;', n_to_remove
        )

        # Remove all elements that old or older
        self.conn.execute('DELETE FROM cache WHERE accesstime <= ?',
                          nth_access_time)
