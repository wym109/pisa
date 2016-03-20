
import os
import time
import re
from collections import OrderedDict
import sqlite3

import cPickle as pickle

from pisa.utils.log import logging, set_verbosity
from pisa.utils import jsons


class MemoryCache(OrderedDict):
    """Simple implementation of a first-in-first-out (FIFO) or least-recently-
    used (LRU) in-memory cache, derived from OrderedDict (i.e., sharing the
    interface of an OrderedDict).

    Parameters
    ----------
    max_depth : int >= 0
        Maximum number of entries in the cache, pruned either by FIFO or LRU
        logic. A `max_depth` of 0 effectively disables caching altogether
        (useful for testing).

    is_lru : bool
        Whether to implement LRU logic (True) or FIFO logic (False). LRU logic
        adds a small computational cost.

    Class attributes
    ----------------
    GLOBAL_CACHE_DEPTH_OVERRIDE : None or int >= 0
        Set to an integer to override the cache depth for *all* memory caches.
        E.g., set this to 0 to disable caching everywhere.

    Based off of code at www.kunxi.org/blog/2014/05/lru-cache-in-python
    """
    GLOBAL_CACHE_DEPTH_OVERRIDE = None
    def __init__(self, max_depth, is_lru=True):
        self.max_depth = max_depth
        self.is_lru = is_lru
        if self.GLOBAL_CACHE_DEPTH_OVERRIDE is not None:
            self.max_depth = self.GLOBAL_CACHE_DEPTH_OVERRIDE
        assert self.max_depth >= 0
        super(MemoryCache, self).__init__()

    def __getitem__(self, key):
        if self.is_lru:
            value = OrderedDict.__getitem__(self, key)
            del self[key]
            OrderedDict.__setitem__(self, key, value)
        else:
            value = OrderedDict.__getitem__(self, key)
        if hasattr(value, 'is_new'):
            value.is_new = False
        return value

    def __setitem__(self, key, value):
        if self.max_depth > 0:
            try:
                self.pop(key)
            except KeyError:
                if len(self) >= self.max_depth:
                    self.popitem(last=False)
            OrderedDict.__setitem__(self, key, value)

    def __delitem__(self, key):
        OrderedDict.__delitem__(self, key)


class DiskCache(object):
    """
    Implements a subset of dict methods but with persistent storage to an on-
    disk sqlite database.

    Parameters
    ----------
    db_fpath : str
        Path to database file; if existing file is specified, schema must match
        that specified by DiskCache.TABLE_SCHEMA.

    row_limit : int
        Limit on the number of rows in the database's table. Pruning is either
        via first-in-first-out (FIFO) or least-recently-used (LRU) logic.

    is_lru : bool
        If True, implement least-recently-used (LRU) logic for removing items
        beyond `row_limit`. This adds an additional ~400 ms to item retrieval
        time.

    Methods
    -------
    __getitem__
    __setitem__
    __delitem__
    __len__
    get
    clear
    keys

    Notes
    -----
    This is not (as of now) thread-safe, but it is multi-process safe. The
    Python sqlite3 api requires a single Python process to have just a single
    connection to a database at a time. Several processes can be safely
    connected to the database simultaneously, however, due to sqlite's locking
    mechanisms that resolve resource contention.

    Access to the database via dict-like syntax:
    >>>> x = {'xyz': [0,1,2,3], 'abc': {'first': (4,5,6)}}
    >>>> disk_cache = DiskCache('/tmp/diskcache.db', row_limit=5, is_lru=False)
    >>>> disk_cache[12] = x
    >>>> y = disk_cache[12]
    >>>> print y == x
    True
    >>>> len(x)
    1
    >>>> x.clear()
    >>>> len(x)
    0
    >>>> # Demonstrating the row-limit:
    >>>> x = [disk_cache.__setitem__(i, 'foo') for i in xrange(10)]
    >>>> len(disk_cache)
    5

    Large databases are slower to work with than small. Therefore it is
    recommended to use separate databases for each stage's cache rather than
    one centralized database acting as the cache for all stages.

    """
    TABLE_SCHEMA = \
        '''CREATE TABLE cache (hash INTEGER PRIMARY KEY,
                               accesstime INTEGER,
                               data BLOB)'''
    def __init__(self, db_fpath, row_limit=100, is_lru=False):
        self.__db_fpath = os.path.expandvars(os.path.expanduser(db_fpath))
        self.__instantiate_db()
        assert 0 < row_limit < 1e6, 'Invalid row_limit:' + str(row_limit)
        self.__row_limit = row_limit
        self.is_lru = is_lru

    def __instantiate_db(self):
        exists = True if os.path.isfile(self.__db_fpath) else False
        conn = self.__connect()
        try:
            if exists:
                # Check that the table format is valid
                sql = ("SELECT sql FROM sqlite_master WHERE type='table' AND"
                       " NAME='cache'")
                cursor = conn.execute(sql)
                schema, = cursor.fetchone()
                # Ignore formatting
                schema = re.sub(r'\s', '', schema).lower()
                ref_schema = re.sub(r'\s', '', self.TABLE_SCHEMA).lower()
                if schema != ref_schema:
                    raise ValueError('Existing database at "%s" has'
                                     'non-matching schema:\n"""%s"""' 
                                     %(self.__db_fpath, schema))
            else:
                # Create the table for storing (hash, data, timestamp) tuples
                conn.execute(self.TABLE_SCHEMA)
                sql = "CREATE INDEX idx0 ON cache(hash)"
                cursor.execute(sql)
                sql = "CREATE INDEX idx1 ON cache(accesstime)"
                conn.execute(sql)
                conn.commit()
        except:
            conn.rollback()
            raise
        finally:
            conn.close()

    def __getitem__(self, hash_val):
        t0 = time.time()
        if not isinstance(hash_val, int):
            raise KeyError('`hash_val` must be int, got "%s"' % type(hash_val))
        conn = self.__connect()
        t1 = time.time();logging.trace('conn: %0.4f' % (t1 - t0))
        try:
            if self.is_lru:
                # Update accesstime
                sql = "UPDATE cache SET accesstime = ? WHERE hash = ?"
                conn.execute(sql, (self.now, hash_val))
                t2 = time.time();logging.trace('update: % 0.4f' % (t2 - t1))
            t2 = time.time()

            # Retrieve contents
            sql = "SELECT data FROM cache WHERE hash = ?"
            cursor = conn.execute(sql, (hash_val,))
            t3 = time.time();logging.trace('select: % 0.4f' % (t3 - t2))
            tmp = cursor.fetchone()
            if tmp is None:
                raise KeyError(str(hash_val))
            data = tmp[0]
            t4 = time.time();logging.trace('fetch: % 0.4f' % (t4 - t3))
            data = pickle.loads(bytes(data))
            t5 = time.time();logging.trace('loads: % 0.4f' % (t5 - t4))
        finally:
            conn.commit()
            conn.close()
            logging.trace('')
        return data

    def __setitem__(self, hash_val, obj):
        t0 = time.time()
        assert isinstance(hash_val, int)
        data = sqlite3.Binary(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
        t1 = time.time();logging.trace('dumps: % 0.4f' % (t1 - t0))

        conn = self.__connect()
        t2 = time.time();logging.trace('conn: % 0.4f' % (t2 - t1))
        try:
            t = time.time()
            sql = "INSERT INTO cache (hash, accesstime, data) VALUES (?, ?, ?)"
            conn.execute(sql, (hash_val, self.now, data))
            t1 = time.time();logging.trace('insert: % 0.4f' % (t1 - t))
            conn.commit()
            t2 = time.time();logging.trace('commit: % 0.4f' % (t2 - t1))

            # Remove oldest-accessed rows in excess of limit
            n_to_remove = len(self) - self.__row_limit
            t3 = time.time();logging.trace('len: % 0.4f' % (t3 - t2))
            if n_to_remove <= 0:
                return

            # Find the access time of the n-th element (sorted ascending by
            # access time)
            sql = ("SELECT accesstime FROM cache ORDER BY accesstime ASC LIMIT"
                   " 1 OFFSET ?")
            cursor = conn.execute(sql, (n_to_remove,))
            t4 = time.time();logging.trace('select old: % 0.4f' % (t4 - t3))
            nth_access_time, = cursor.fetchone()
            t5 = time.time();logging.trace('fetch: % 0.4f' % (t5 - t4))

            # Remove all elements that old or older
            sql = "DELETE FROM cache WHERE accesstime < ?"
            conn.execute(sql, (nth_access_time,))
            t6 = time.time();logging.trace('delete: % 0.4f' % (t6 - t5))
        except:
            t = time.time()
            conn.rollback()
            logging.trace('rollback: % 0.4f' % (time.time() - t))
            raise
        else:
            t = time.time()
            conn.commit()
            logging.trace('commit: % 0.4f' % (time.time() - t))
        finally:
            t = time.time()
            conn.close()
            logging.trace('close: % 0.4f' % (time.time() - t))
            logging.trace('')

    def __delitem__(self, hash_val):
        conn = self.__connect()
        try:
            sql = "DELETE FROM cache WHERE hash = ?"
            cursor = conn.execute(sql, (hash_val,))
        except:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def __len__(self):
        conn = self.__connect()
        try:
            cursor = conn.execute('SELECT COUNT (*) FROM cache')
            count, = cursor.fetchone()
        finally:
            conn.close()
        return count

    def get(self, hash_val, dflt=None):
        rslt = dflt
        try:
            rslt = self.__getitem__(hash_val)
        except:
            pass
        return rslt

    def clear(self):
        conn = self.__connect()
        try:
            conn.execute('DELETE FROM cache')
        except:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            conn.close()

    def keys(self):
        conn = self.__connect()
        try:
            sql = "SELECT hash FROM cache ORDER BY accesstime ASC"
            cursor = conn.execute(sql)
            k = [k[0] for k in cursor.fetchall()]
        finally:
            conn.close()
        return k

    def __connect(self):
        conn = sqlite3.connect(
            self.__db_fpath,
            isolation_level=None, check_same_thread=False, timeout=10,
        )

        # Trust journaling to memory
        sql = "PRAGMA journal_mode=MEMORY"
        conn.execute(sql)

        # Trust OS to complete transaction
        sql = "PRAGMA synchronous=0"
        conn.execute(sql)

        # Make file size shrink when items are removed
        sql = "PRAGMA auto_vacuum=FULL"
        conn.execute(sql)

        return conn

    @property
    def now(self):
        """Microseconds since the epoch"""
        return int(time.time() * 1e6)

