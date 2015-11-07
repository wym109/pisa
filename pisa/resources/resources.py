#
# resources.py
#
# Tools to obtain one of the input resources files needed for PISA.
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-06-10

import os, sys
from pisa.utils.log import logging
from pkg_resources import resource_filename


def find_resource(filename, fail=True):
    '''
    Try to find the resource given by directory/filename. Will first check if
    filename is an absolute path, then relative to the $PISA
    environment variable if set. Otherwise will look in the resources directory
    of the pisa installation. Will return the file handle or throw an Exception
    if the file is not found.
    '''

    # First check for absolute path
    fpath = os.path.expandvars(os.path.expanduser(filename))
    logging.trace("Checking if %s is a file..." % fpath)
    if os.path.isfile(fpath):
        logging.debug('Found %s' % (fpath))
        return fpath

    # Next check if $PISA is set in environment
    logging.trace("Checking environment for $PISA...")
    if 'PISA' in os.environ:
        rpath = os.path.expandvars(os.path.expanduser(os.environ['PISA']))
        logging.debug('Searching resource path PISA=%s' % rpath)

        fpath = os.path.join(rpath, filename)
        if os.path.isfile(fpath):
            logging.debug('Found %s at %s' % (filename, fpath))
            return fpath

    # Not in the resource path, so look inside the package
    logging.trace('Searching package resources...')
    fpath = resource_filename(__name__, filename)
    if os.path.isfile(fpath):
        logging.debug('Found %s at %s' % (filename, fpath))
        return fpath

    # Nowhere to be found
    if fail:
        raise IOError('Could not find resource "%s"' % filename)
    else:
        logging.debug('Could not find resource "%s"' % filename)
        return None


def open_resource(filename):
    '''
    Find the resource file (see find_resource), open it and return a file
    handle.
    '''
    try:
        return open(find_resource(filename))
    except (IOError, OSError), e:
        logging.error('Unable to open resource "%s"' % filename)
        logging.error(e)
        sys.exit(1)
