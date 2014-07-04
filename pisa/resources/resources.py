#
# resources.py
#
# Tools to obtain one of the input resources files needed for PISA. 
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-06-10

import os
import logging
from pkg_resources import resource_stream, resource_filename


def find_resource(filename):
    '''
    Try to find the resource given by directory/filename. Will first check if
    filename is an absolute path, then relative to the $PISA
    environment variable if set. Otherwise will look in the resources directory
    of the pisa installation. Will return the file handle or throw an Exception
    if the file is not found.
    '''

    #First check for absolute path
    fpath = os.path.expanduser(os.path.expandvars(filename))
    if os.path.isfile(fpath):
        logging.debug('Found %s'%(fpath))
        return fpath
    
    #Next check if $PISA is set in environment
    if 'PISA' in os.environ:
        rpath = os.path.expanduser(os.path.expandvars(os.environ['PISA']))
        logging.debug('Searching resource path PISA=%s'%rpath)

        fpath = os.path.join(rpath,filename)
        if os.path.isfile(fpath):
            logging.debug('Found %s at %s'%(filename,fpath))
            return fpath

    #Not in the resource path, so look inside the package
    logging.debug('Searching package resources...')
    fpath = resource_filename(__name__,filename)
    if os.path.isfile(fpath):
        logging.debug('Found %s at %s'%(filename,fpath))
        return fpath

    #Nowhere to be found
    raise IOError('Could not find resource "%s"'%filename)


def open_resource(filename):
    '''
    Find the resource file (see find_resource), open it and return a file
    handle.
    '''
    try:
        return open(find_resource(filename))
    except (IOError, OSError), e:
        logging.error('Unable to open resource "%s"'%filename)
        logging.error(e)
        sys.exit(1)
