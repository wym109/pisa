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


def find_resource(resourcename, is_dir=False, fail=True):
    """Try to find a resource or directory.

    First check if resourcename is an absolute path, then relative to the $PISA
    environment variable if it is set. Otherwise, look in the resources
    directory of the pisa installation.

    Parameters
    ----------
    resourcename : str
        File or directory name to locate. The `resourcename` can include an
        absolute or relative (to $PWD or within the $PISA/resources directory)
        path. E.g., 'directory/resourcename'.

    is_dir : bool
        Whether to return a directory or file

    fail : bool
        If True, raise IOError if resourcename not found
        If False, return None if resourcename not found

    Returns
    -------
    If found, File or directory path; if not found and `fail` is False, returns None.

    Raises
    ------
    Exception if the file is not found and `fail=True`.
    """

    if is_dir:
        isentity = os.path.isdir
        entity_type = 'dir'
    else:
        isentity = os.path.isfile
        entity_type = 'file'

    # First check for absolute path
    rsrc_path = os.path.expandvars(os.path.expanduser(resourcename))
    logging.trace("Checking if %s is a %s..." % (rsrc_path, entity_type))
    if isentity(rsrc_path):
        logging.debug('Found %s' % (rsrc_path))
        return rsrc_path

    # Next check if $PISA is set in environment
    logging.trace("Checking environment for $PISA...")
    if 'PISA' in os.environ:
        rpath = os.path.expandvars(os.path.expanduser(os.environ['PISA']))
        logging.debug('Searching resource path PISA=%s' % rpath)

        rsrc_path = os.path.join(rpath, resourcename)
        if isentity(rsrc_path):
            logging.debug('Found %s at %s' % (resourcename, rsrc_path))
            return rsrc_path

    # Not in the resource path, so look inside the package
    logging.trace('Searching package resources...')
    rsrc_path = resource_filename(__name__, resourcename)
    if isentity(rsrc_path):
        logging.debug('Found %s at %s' % (resourcename, rsrc_path))
        return rsrc_path

    # Nowhere to be found
    if fail:
        raise IOError('Could not find %s resource "%s"' % (entity_type,
                                                           resourcename))
    logging.debug('Could not find %s resource "%s"' % (entity_type,
                                                       resourcename))
    return None


def open_resource(filename, mode='r'):
    """Find the resource file (see find_resource), open it and return a file
    handle.
    """
    return open(find_resource(filename, fail=True), mode=mode)

